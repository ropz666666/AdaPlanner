import os
import json
import argparse
import csv
import re
from typing import List, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import ssl
import json as _json

def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_paths_only(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        s = f.read()
    i = s.find('"paths"')
    if i == -1:
        return {}
    j = s.find('{', i)
    if j == -1:
        return {}
    cnt = 0
    end = None
    for k in range(j, len(s)):
        c = s[k]
        if c == '{':
            cnt += 1
        elif c == '}':
            cnt -= 1
            if cnt == 0:
                end = k + 1
                break
    if end is None:
        return {}
    sub = s[j:end]
    try:
        return json.loads(sub)
    except Exception:
        return {}

def list_endpoints(oas: dict) -> List[str]:
    eps = []
    paths = oas.get('paths', {})
    for p, methods in paths.items():
        for m in methods.keys():
            if m.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                eps.append(f"{m.upper()} {p}")
    return eps

def build_solution_prompt(query: str, endpoints: List[str]) -> str:
    ep_text = '\n'.join(endpoints)
    head = (
        "You are an API planner. Given a requirement and available endpoints, plan a minimal ordered sequence of calls. "
        "For each step, mark with '[Step xx]' and a one-sentence reason. After planning, output only the calls under a '[Plan]' section, one per line, format 'METHOD /path'."
    )
    return f"{head}\n\nRequirement:\n{query}\n\nAvailable endpoints:\n{ep_text}\n\n[Plan]"

def build_check_prompt(plan_text: str, endpoints: List[str]) -> str:
    ep_text = '\n'.join(endpoints)
    return (
        "Validate the plan lines below against the available endpoints. "
        "Respond with '[Decision]: Yes' if all lines are valid, else '[Decision]: No'. "
        "If 'No', provide a corrected plan under '[Revised plan]' using only endpoints provided.\n\n"
        f"Plan:\n{plan_text}\n\nEndpoints:\n{ep_text}\n"
    )

def build_fix_prompt(query: str, error_msg: str, endpoints: List[str]) -> str:
    ep_text = '\n'.join(endpoints)
    return (
        "Revise the plan to address the error below. Output only corrected calls under '[Revised plan]'. "
        f"Requirement:\n{query}\nError:\n{error_msg}\n\nEndpoints:\n{ep_text}\n"
    )

def parse_calls(text: str) -> List[str]:
    cleaned = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).strip('`'), text)
    lines = [l.strip() for l in cleaned.splitlines()]
    calls = []
    pat = re.compile(r"^(GET|POST|PUT|DELETE|PATCH)\s+(/[^\s]+)")
    for l in lines:
        m = pat.match(l)
        if m:
            calls.append(f"{m.group(1)} {m.group(2)}")
    return calls

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def write_csv_header(path: str):
    ensure_dir(path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['requirement', 'pass_rate', 'success_rate', 'llm_calls', 'token_cost', 'result', 'error'])

def evaluate(pred: List[str], gt: List[str]) -> Tuple[float, float]:
    ok = 1.0 if pred == gt else 0.0
    return ok, ok

def run(dataset_path: str, oas_path: str, output_csv: str, use_llm: bool, limit: int, start: int, closed_loop: bool, num_try: int, dataset: str):
    data = load_json(dataset_path)
    try:
        oas = load_json(oas_path)
        endpoints = list_endpoints(oas)
    except Exception:
        paths = load_paths_only(oas_path)
        endpoints = list_endpoints({'paths': paths}) if paths else []
    write_csv_header(output_csv)
    llm_calls_total = 0
    token_cost_total = 0.0
    rows = []
    subset = data[start:start+limit] if limit else data[start:]
    for i, item in enumerate(subset):
        query = item['query']
        error = ''
        plan = plan_calls(query, endpoints, use_llm)
        llm_calls = 0
        if use_llm:
            llm_calls += 2
        pass_rate = 0.0
        success_rate = 0.0
        if closed_loop and plan:
            tries = num_try
            for attempt in range(tries):
                executed = execute_calls(dataset, query, endpoints, plan)
                ok = len(executed)
                total = len(plan)
                pass_rate = (ok / total) if total else 0.0
                success_rate = pass_rate
                if pass_rate >= 1.0:
                    break
                if use_llm:
                    try:
                        import openai
                        api_key = os.getenv('OPENAI_API_KEY', '')
                        openai.api_key = api_key
                        err_txt = f'Executed {ok}/{total}.'
                        p_fix = build_fix_prompt(query, err_txt, endpoints)
                        r_fix = openai.ChatCompletion.create(model='gpt-4o', messages=[{'role':'user','content':p_fix}], temperature=0)
                        t_fix = r_fix['choices'][0]['message']['content']
                        llm_calls += 1
                        revised = parse_calls(t_fix)
                        plan = revised if revised else plan
                    except Exception as e:
                        error = str(e)
                        break
        rows.append([query, pass_rate, success_rate, llm_calls, token_cost_total, json.dumps(plan, ensure_ascii=False), error])
    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)

def _http_get(url: str, headers: dict, params: dict) -> dict:
    qs = urlencode(params or {})
    full = f"{url}?{qs}" if qs else url
    req = Request(full, headers=headers, method='GET')
    ctx = ssl.create_default_context()
    with urlopen(req, context=ctx) as resp:
        data = resp.read().decode('utf-8')
        try:
            return _json.loads(data)
        except Exception:
            return {'raw': data}

def plan_calls(query: str, endpoints: List[str], use_llm: bool) -> List[str]:
    if use_llm:
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY', '')
            openai.api_key = api_key
            p1 = build_solution_prompt(query, endpoints)
            r1 = openai.ChatCompletion.create(model='gpt-4o', messages=[{'role':'user','content':p1}], temperature=0)
            t1 = r1['choices'][0]['message']['content']
            plan_lines = parse_calls(t1)
            p2 = build_check_prompt("\n".join(plan_lines), endpoints)
            r2 = openai.ChatCompletion.create(model='gpt-4o', messages=[{'role':'user','content':p2}], temperature=0)
            t2 = r2['choices'][0]['message']['content']
            if '[Decision]: Yes' in t2:
                return plan_lines
            revised = parse_calls(t2)
            return revised if revised else plan_lines
        except Exception:
            pass
    return rule_plan_calls(query, endpoints)

def rule_plan_calls(query: str, endpoints: List[str]) -> List[str]:
    q = query.lower()
    def has(prefix: str):
        return any(e.startswith('GET ' + prefix) for e in endpoints)
    calls: List[str] = []
    base_added = False
    if 'top' in q and 'rated' in q and has('/movie/top_rated'):
        calls.append('GET /movie/top_rated')
        base_added = True
    elif 'popular' in q:
        if 'tv' in q and has('/tv/popular'):
            calls.append('GET /tv/popular')
            base_added = True
        elif has('/movie/popular'):
            calls.append('GET /movie/popular')
            base_added = True
    elif 'trending' in q:
        mt = 'tv' if 'tv' in q else 'movie'
        pref = f'/trending/{mt}/day'
        if has(pref):
            calls.append('GET ' + pref)
            base_added = True
    elif 'on the air' in q and has('/tv/on_the_air'):
        calls.append('GET /tv/on_the_air')
        base_added = True
    elif 'collection' in q and has('/search/collection'):
        calls.append('GET /search/collection')
        base_added = True
    elif 'company' in q and has('/search/company'):
        calls.append('GET /search/company')
        base_added = True
    elif 'person' in q or 'actor' in q or 'director' in q:
        if has('/search/person'):
            calls.append('GET /search/person')
            base_added = True
    else:
        if 'tv' in q and has('/search/tv'):
            calls.append('GET /search/tv')
            base_added = True
        elif has('/search/movie'):
            calls.append('GET /search/movie')
            base_added = True
    if ('director' in q or 'lead actor' in q or 'cast' in q) and has('/movie/{movie_id}/credits'):
        calls.append('GET /movie/{movie_id}/credits')
    if ('reviews' in q) and has('/movie/{movie_id}/reviews'):
        calls.append('GET /movie/{movie_id}/reviews')
    if ('keyword' in q or 'keywords' in q) and has('/movie/{movie_id}/keywords'):
        calls.append('GET /movie/{movie_id}/keywords')
    if ('image' in q or 'poster' in q or 'photo' in q) and has('/movie/{movie_id}/images'):
        calls.append('GET /movie/{movie_id}/images')
    if ('release date' in q or 'released' in q) and has('/movie/{movie_id}/release_dates'):
        calls.append('GET /movie/{movie_id}/release_dates')
    if ('similar' in q) and has('/movie/{movie_id}/similar'):
        calls.append('GET /movie/{movie_id}/similar')
    if ('recommend' in q or 'recommendations' in q):
        if 'tv' in q and has('/tv/{tv_id}/recommendations'):
            calls.append('GET /tv/{tv_id}/recommendations')
        elif has('/movie/{movie_id}/recommendations'):
            calls.append('GET /movie/{movie_id}/recommendations')
    if 'collection' in q and has('/collection/{collection_id}/images'):
        calls.append('GET /collection/{collection_id}/images')
    return calls if calls else (['GET /search/movie'] if has('/search/movie') else [])

def execute_calls(dataset: str, query: str, endpoints: List[str], calls: List[str]) -> List[dict]:
    out = []
    ctx = {}
    if dataset == 'tmdb':
        base = 'https://api.themoviedb.org/3'
        api_key = os.getenv('TMDB_API_KEY', '')
        if not api_key:
            return out
        for c in calls:
            m, p = c.split(' ', 1)
            if m != 'GET':
                continue
            params = {'api_key': api_key}
            if p.startswith('/search/movie'):
                params['query'] = query
                res = _http_get(base + '/search/movie', {}, params)
                out.append({'call': c, 'response': res})
                try:
                    ctx['movie_id'] = res['results'][0]['id']
                except Exception:
                    pass
            elif p.startswith('/search/person'):
                params['query'] = query
                res = _http_get(base + '/search/person', {}, params)
                out.append({'call': c, 'response': res})
                try:
                    ctx['person_id'] = res['results'][0]['id']
                except Exception:
                    pass
            elif p.startswith('/search/tv'):
                params['query'] = query
                res = _http_get(base + '/search/tv', {}, params)
                out.append({'call': c, 'response': res})
                try:
                    ctx['tv_id'] = res['results'][0]['id']
                except Exception:
                    pass
            elif p.startswith('/movie/top_rated'):
                res = _http_get(base + '/movie/top_rated', {}, params)
                out.append({'call': c, 'response': res})
                try:
                    ctx['movie_id'] = res['results'][0]['id']
                except Exception:
                    pass
            elif p.startswith('/trending/'):
                parts = p.split('/')
                media_type = parts[2] if len(parts) > 2 else 'movie'
                time_window = parts[3] if len(parts) > 3 else 'day'
                res = _http_get(base + f'/trending/{media_type}/{time_window}', {}, params)
                out.append({'call': c, 'response': res})
                try:
                    ctx['movie_id'] = res['results'][0]['id']
                except Exception:
                    pass
            elif p.startswith('/movie/') and 'credits' in p:
                mid = ctx.get('movie_id')
                if not mid:
                    continue
                res = _http_get(base + f'/movie/{mid}/credits', {}, params)
                out.append({'call': c, 'response': res})
            elif p.startswith('/movie/') and 'keywords' in p:
                mid = ctx.get('movie_id')
                if not mid:
                    continue
                res = _http_get(base + f'/movie/{mid}/keywords', {}, params)
                out.append({'call': c, 'response': res})
            elif p.startswith('/movie/') and 'images' in p:
                mid = ctx.get('movie_id')
                if not mid:
                    continue
                res = _http_get(base + f'/movie/{mid}/images', {}, params)
                out.append({'call': c, 'response': res})
            elif p.startswith('/movie/latest'):
                res = _http_get(base + '/movie/latest', {}, params)
                out.append({'call': c, 'response': res})
                try:
                    ctx['movie_id'] = res.get('id')
                except Exception:
                    pass
            elif p.startswith('/person/') and 'movie_credits' in p:
                pid = ctx.get('person_id')
                if not pid:
                    continue
                res = _http_get(base + f'/person/{pid}/movie_credits', {}, params)
                out.append({'call': c, 'response': res})
            elif p.startswith('/person/') and 'tv_credits' in p:
                pid = ctx.get('person_id')
                if not pid:
                    continue
                res = _http_get(base + f'/person/{pid}/tv_credits', {}, params)
                out.append({'call': c, 'response': res})
            elif p.startswith('/tv/') and 'recommendations' in p:
                tid = ctx.get('tv_id')
                if not tid:
                    continue
                res = _http_get(base + f'/tv/{tid}/recommendations', {}, params)
                out.append({'call': c, 'response': res})
            elif p.startswith('/tv/on_the_air'):
                res = _http_get(base + '/tv/on_the_air', {}, params)
                out.append({'call': c, 'response': res})
            else:
                res = _http_get(base + p, {}, params)
                out.append({'call': c, 'response': res})
    elif dataset == 'spotify':
        base = 'https://api.spotify.com/v1'
        token = os.getenv('SPOTIFY_TOKEN', '')
        if not token:
            return out
        headers = {'Authorization': f'Bearer {token}'}
        for idx, c in enumerate(calls):
            m, p = c.split(' ', 1)
            if m != 'GET':
                continue
            params = {}
            if p.startswith('/search'):
                nxt = calls[idx + 1] if idx + 1 < len(calls) else ''
                t = 'track'
                if '/artists/' in nxt:
                    t = 'artist'
                elif '/albums/' in nxt:
                    t = 'album'
                params = {'q': query, 'type': t, 'limit': 1}
                res = _http_get(base + '/search', headers, params)
                out.append({'call': c, 'response': res})
                try:
                    if t == 'artist':
                        ctx['artist_id'] = res['artists']['items'][0]['id']
                    elif t == 'album':
                        ctx['album_id'] = res['albums']['items'][0]['id']
                    else:
                        it = res['tracks']['items'][0]
                        ctx['track_id'] = it['id']
                        ctx['track_uri'] = it['uri']
                except Exception:
                    pass
            elif p.startswith('/artists/') and '/albums' in p:
                aid = ctx.get('artist_id')
                if not aid:
                    continue
                res = _http_get(base + f'/artists/{aid}/albums', headers, {'limit': 1})
                out.append({'call': c, 'response': res})
                try:
                    ctx['album_id'] = res['items'][0]['id']
                except Exception:
                    pass
            elif p.startswith('/albums/') and '/tracks' in p:
                alid = ctx.get('album_id')
                if not alid:
                    continue
                res = _http_get(base + f'/albums/{alid}/tracks', headers, {'limit': 1})
                out.append({'call': c, 'response': res})
                try:
                    it = res['items'][0]
                    ctx['track_id'] = it['id']
                    ctx['track_uri'] = it.get('uri') or f"spotify:track:{it['id']}"
                except Exception:
                    pass
            elif p.startswith('/me'):
                res = _http_get(base + '/me', headers, {})
                out.append({'call': c, 'response': res})
                try:
                    ctx['user_id'] = res['id']
                except Exception:
                    pass
            else:
                res = _http_get(base + p, headers, {})
                out.append({'call': c, 'response': res})
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['spotify', 'tmdb'], required=True)
    parser.add_argument('--use-llm', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--query', type=str, default='')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--out', type=str, default='')
    parser.add_argument('--closed-loop', action='store_true')
    parser.add_argument('--num-try', type=int, default=3)
    args = parser.parse_args()
    base = os.path.dirname(os.path.dirname(__file__))
    ds_path = os.path.join(base, 'experiment', 'datasets', f"{args.dataset}.json")
    oas_path = os.path.join(base, 'api_doc', f"{args.dataset}_oas.json")
    out_csv_default = os.path.join(base, 'experiment', 'result', f"{args.dataset}.csv")
    out_csv = args.out if args.out else out_csv_default
    run(ds_path, oas_path, out_csv, args.use_llm, args.limit, args.start, args.closed_loop, args.num_try, args.dataset)
    if args.execute:
        try:
            oas = load_json(oas_path)
            endpoints = list_endpoints(oas)
        except Exception:
            paths = load_paths_only(oas_path)
            endpoints = list_endpoints({'paths': paths}) if paths else []
        data = []
        if args.query:
            data = [{'query': args.query}]
        else:
            data = load_json(ds_path)
            data = data[args.start:args.start+args.limit] if args.limit else data[args.start:]
        for item in data:
            query = item['query']
            calls = plan_calls(query, endpoints, args.use_llm)
            resp = execute_calls(args.dataset, query, endpoints, calls)
            out_jsonl = os.path.join(base, 'experiment', 'result', f"{args.dataset}_responses.jsonl")
            ensure_dir(out_jsonl)
            with open(out_jsonl, 'a', encoding='utf-8') as jf:
                for r in resp:
                    jf.write(_json.dumps({'requirement': query, **r}, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()

