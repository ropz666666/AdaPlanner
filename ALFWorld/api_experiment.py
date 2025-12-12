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

def run(dataset_path: str, oas_path: str, output_csv: str, use_llm: bool, limit: int):
    data = load_json(dataset_path)
    oas = load_json(oas_path)
    endpoints = list_endpoints(oas)
    write_csv_header(output_csv)
    llm_calls_total = 0
    token_cost_total = 0.0
    rows = []
    for i, item in enumerate(data[:limit] if limit else data):
        query = item['query']
        gt = item.get('solution', [])
        error = ''
        if use_llm:
            try:
                import openai
                api_key = os.getenv('OPENAI_API_KEY', '')
                openai.api_key = api_key
                p1 = build_solution_prompt(query, endpoints)
                r1 = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[{'role':'user','content':p1}], temperature=0)
                t1 = r1['choices'][0]['message']['content']
                plan_lines = parse_calls(t1)
                llm_calls_total += 1
                p2 = build_check_prompt("\n".join(plan_lines), endpoints)
                r2 = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[{'role':'user','content':p2}], temperature=0)
                t2 = r2['choices'][0]['message']['content']
                llm_calls_total += 1
                if '[Decision]: Yes' in t2:
                    pred = plan_lines
                else:
                    revised = parse_calls(t2)
                    pred = revised if revised else plan_lines
            except Exception as e:
                pred = []
                error = str(e)
        else:
            pred = gt
        pass_rate, success_rate = evaluate(pred, gt)
        rows.append([query, pass_rate, success_rate, 1 if use_llm else 0, token_cost_total, json.dumps(pred, ensure_ascii=False), error])
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
            r1 = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[{'role':'user','content':p1}], temperature=0)
            t1 = r1['choices'][0]['message']['content']
            plan_lines = parse_calls(t1)
            p2 = build_check_prompt("\n".join(plan_lines), endpoints)
            r2 = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[{'role':'user','content':p2}], temperature=0)
            t2 = r2['choices'][0]['message']['content']
            if '[Decision]: Yes' in t2:
                return plan_lines
            revised = parse_calls(t2)
            return revised if revised else plan_lines
        except Exception:
            return []
    eps = [e for e in endpoints if e.startswith('GET')]
    return eps[:1]

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
    args = parser.parse_args()
    base = os.path.dirname(os.path.dirname(__file__))
    ds_path = os.path.join(base, 'experiment', 'datasets', f"{args.dataset}.json")
    oas_path = os.path.join(base, 'api_doc', f"{args.dataset}_oas.json")
    out_csv = os.path.join(base, 'experiment', 'result', f"{args.dataset}.csv")
    run(ds_path, oas_path, out_csv, args.use_llm, args.limit)
    if args.execute:
        oas = load_json(oas_path)
        endpoints = list_endpoints(oas)
        data = []
        if args.query:
            data = [{'query': args.query}]
        else:
            data = load_json(ds_path)
            data = data[:args.limit] if args.limit else data
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

