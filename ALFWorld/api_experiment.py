import os
import json
import argparse
import csv
import re
from typing import List, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import ssl
import json as _json
import time

def load_simple_yaml(path: str) -> dict:
    out = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                if ':' in s:
                    k, v = s.split(':', 1)
                    k = k.strip()
                    v = v.strip()
                    if v.startswith('"') and v.endswith('"'):
                        v = v[1:-1]
                    if v.startswith("'") and v.endswith("'"):
                        v = v[1:-1]
                    if k and v is not None:
                        out[k] = v
    except Exception:
        pass
    return out

def total_tokens_of(resp) -> int:
    try:
        u = getattr(resp, 'usage', None)
        if u is None:
            return 0
        if isinstance(u, dict):
            v = u.get('total_tokens', 0)
            return int(v or 0)
        v = getattr(u, 'total_tokens', 0)
        return int(v or 0)
    except Exception:
        return 0

def is_success(resp: dict) -> bool:
    try:
        if resp is None:
            return False
        if isinstance(resp, dict):
            if 'http_error_code' in resp or 'url_error' in resp:
                return False
        return True
    except Exception:
        return False

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

def load_paths_text(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            s = f.read()
        i = s.find('"paths"')
        if i == -1:
            return ''
        j = s.find('{', i)
        if j == -1:
            return ''
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
            return ''
        return s[j:end]
    except Exception:
        return ''

def list_endpoints_from_text(paths_text: str) -> List[str]:
    eps: List[str] = []
    if not paths_text:
        return eps
    try:
        # Roughly split by path blocks
        for m in re.finditer(r'"(/[^\"]+)"\s*:\s*\{([\s\S]*?)\}', paths_text):
            p = m.group(1)
            block = m.group(2)
            for meth in ['get', 'post', 'put', 'delete', 'patch']:
                if re.search(r'\b' + meth + '\b', block, re.IGNORECASE):
                    eps.append(meth.upper() + ' ' + p)
        return eps
    except Exception:
        return eps

def list_endpoints(oas: dict) -> List[str]:
    eps = []
    paths = oas.get('paths', {})
    for p, methods in paths.items():
        for m in methods.keys():
            if m.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                eps.append(f"{m.upper()} {p}")
    return eps

def spotify_known_endpoints() -> List[str]:
    return [
        'GET /search',
        'GET /artists/{artist_id}/albums',
        'GET /artists/{artist_id}/top-tracks',
        'GET /albums/{album_id}/tracks',
        'GET /me',
        'GET /me/playlists',
        'GET /me/player/currently-playing',
        'GET /me/player/recently-played',
        'POST /users/{user_id}/playlists',
        'POST /playlists',
        'POST /playlists/{playlist_id}/tracks',
        'POST /me/player/queue',
        'POST /me/player/next',
        'POST /me/player/previous',
        'PUT /me/player/volume',
        'PUT /me/following',
        'PUT /me/albums',
        'PUT /me/player/play',
        'PUT /me/player/pause',
        'PUT /me/player/shuffle',
        'PUT /me/player/repeat'
    ]

def spotify_supported_endpoints() -> List[str]:
    return [
        'GET /search',
        'GET /artists/{artist_id}/albums',
        'GET /artists/{artist_id}/top-tracks',
        'GET /albums/{album_id}/tracks',
        'GET /me',
        'POST /users/{user_id}/playlists',
        'POST /playlists',
        'POST /playlists/{playlist_id}/tracks',
        'POST /me/player/queue',
        'POST /me/player/next',
        'PUT /me/player/volume',
        'PUT /me/albums',
        'PUT /me/following'
    ]

def build_solution_prompt(query: str, endpoints: List[str], oas_text: str = '') -> str:
    ep_text = '\n'.join(endpoints)
    head = (
        "You are an API planner. Given a requirement and available endpoints, plan a minimal ordered sequence of calls. "
        "For each step, mark with '[Step xx]' and a one-sentence reason. After planning, output only the calls under a '[Plan]' section, one per line, format 'METHOD /path'."
    )
    if oas_text:
        return f"{head}\n\nRequirement:\n{query}\n\nAvailable endpoints:\n{ep_text}\n\nOAS paths:\n{oas_text}\n\n[Plan]"
    return f"{head}\n\nRequirement:\n{query}\n\nAvailable endpoints:\n{ep_text}\n\n[Plan]"

def build_check_prompt(plan_text: str, endpoints: List[str], oas_text: str = '') -> str:
    ep_text = '\n'.join(endpoints)
    base = (
        "Validate the plan lines below against the available endpoints. "
        "Respond with '[Decision]: Yes' if all lines are valid, else '[Decision]: No'. "
        "If 'No', provide a corrected plan under '[Revised plan]' using only endpoints provided.\n\n"
        f"Plan:\n{plan_text}\n\nEndpoints:\n{ep_text}\n"
    )
    if oas_text:
        return base + f"\nOAS paths:\n{oas_text}\n"
    return base

def build_fix_prompt(query: str, error_msg: str, endpoints: List[str], oas_text: str = '') -> str:
    ep_text = '\n'.join(endpoints)
    base = (
        "Revise the plan to address the error below. Output only corrected calls under '[Revised plan]'. "
        f"Requirement:\n{query}\nError:\n{error_msg}\n\nEndpoints:\n{ep_text}\n"
    )
    if oas_text:
        return base + f"\nOAS paths:\n{oas_text}\n"
    return base

def build_answer_prompt(query: str, executed: List[dict]) -> str:
    parts = []
    for it in executed:
        call = it.get('call', '')
        resp = it.get('response', {})
        try:
            snippet = json.dumps(resp, ensure_ascii=False)
        except Exception:
            snippet = str(resp)
        if len(snippet) > 40000:
            snippet = snippet[:40000]
        parts.append(f"[Call] {call}\n[Data] {snippet}")
    data_block = "\n\n".join(parts)
    return (
        "You are a tool-using assistant. Derive the final answer using ONLY the provided API data. "
        "Cite exact fields if necessary, and be concise. If data is insufficient, say 'Insufficient data'.\n\n"
        f"Requirement:\n{query}\n\nAPI Data:\n{data_block}\n\nAnswer:"
    )

def summarize_spotify_result(query: str, executed: List[dict]) -> str:
    def field_kvs(d: dict) -> List[str]:
        keys = ['id', 'name', 'snapshot_id', 'uri', 'href', 'status', 'total']
        kv = []
        for k in keys:
            v = d.get(k)
            if v is not None:
                kv.append(f"{k}={v}")
        if 'http_error_code' in d:
            kv.append(f"http_error_code={d.get('http_error_code')}")
        if 'error' in d and isinstance(d.get('error'), str):
            kv.append(f"error={d.get('error')[:120]}")
        return kv
    lines: List[str] = []
    playlist_name = extract_query_value(query, 'playlist') or None
    seen = set()
    for it in executed:
        call = it.get('call', '')
        resp = it.get('response', {}) or {}
        req = it.get('request', {}) or {}
        ok = is_success(resp)
        prefix = 'OK' if ok else 'ERR'
        info = []
        # High-level message
        if 'POST /playlists' in call or ('/users/' in call and '/playlists' in call and call.startswith('POST')):
            name = resp.get('name') or playlist_name or 'playlist'
            pid = resp.get('id')
            sig = f"create:{pid}"
            if sig in seen:
                continue
            seen.add(sig)
            info.append(f"Created playlist '{name}' id {pid}")
        elif '/playlists/' in call and '/tracks' in call and call.startswith('POST'):
            uris = (req.get('body', {}).get('uris') if isinstance(req, dict) else None) or []
            cnt = len(uris)
            info.append(f"Added {cnt} tracks to playlist")
        elif '/me/player/queue' in call and call.startswith('POST'):
            uri = (req.get('params', {}).get('uri') if isinstance(req, dict) else None)
            info.append(f"Queued track {uri}")
        elif '/me/player/next' in call and call.startswith('POST'):
            info.append('Skipped to next track')
        elif '/me/player/volume' in call and call.startswith('PUT'):
            vp = (req.get('params', {}).get('volume_percent') if isinstance(req, dict) else None)
            info.append(f"Set volume to {vp}")
        elif '/me/following' in call and call.startswith('PUT'):
            info.append('Followed artist')
        elif '/me/albums' in call and call.startswith('PUT'):
            info.append('Saved album to library')
        # Append compact response fields and any error
        kvs = field_kvs(resp)
        if req:
            # include request params/body snapshot
            params = req.get('params') or {}
            body = req.get('body') or {}
            if params:
                kvs.append('params=' + _json.dumps(params, ensure_ascii=False)[:120])
            if body:
                kvs.append('body=' + _json.dumps(body, ensure_ascii=False)[:120])
        line = f"[{prefix}] {call} -> " + (', '.join(info + kvs) if (info or kvs) else 'no details')
        lines.append(line)
    return '\n'.join(lines) if lines else 'Operation attempted; see responses for details.'

def responses_text(executed: List[dict]) -> str:
    parts: List[str] = []
    for it in executed:
        resp = it.get('response', {})
        try:
            s = json.dumps(resp, ensure_ascii=False)
        except Exception:
            s = str(resp)
        parts.append(s)
    return '\n'.join(parts)

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

def extract_query_value(query: str, kind: str) -> str:
    q = query.strip()
    m = re.findall(r'"([^"]+)"|\'([^\']+)\'', q)
    if m:
        for grp in m:
            cand = grp[0] or grp[1]
            if cand:
                return cand.strip()
    phrases = re.findall(r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)', q)
    if phrases:
        if kind == 'person':
            for p in phrases[::-1]:
                if len(p.split()) >= 2 and not p.startswith('The '):
                    return p
        if kind in ['movie', 'tv', 'collection', 'company', 'network']:
            for p in phrases:
                return p
    words = re.findall(r'[A-Za-z]+', q)
    return ' '.join(words[-2:]) if words else ''

def extract_number(query: str, default: int = 1) -> int:
    q = query.lower()
    m = re.search(r'(\d+)', q)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    word_map = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    for w, n in word_map.items():
        if w in q:
            return n
    return default

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def write_csv_header(path: str):
    ensure_dir(path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['requirement', 'pass_rate', 'success_rate', 'llm_calls', 'token_cost', 'result', 'error', 'cost_time'])

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
        try:
            endpoints = list_endpoints({'paths': paths}) if paths else []
        except Exception:
            endpoints = []
        if not endpoints:
            paths_text = load_paths_text(oas_path)
            endpoints = list_endpoints_from_text(paths_text)
    oas_paths_text = load_paths_text(oas_path)
    if dataset == 'spotify':
        endpoints = spotify_supported_endpoints()
    write_csv_header(output_csv)
    rows = []
    subset = data[start:start+limit] if limit else data[start:]
    for i, item in enumerate(subset):
        t0 = time.time()
        query = item.get('query', '')
        error = ''
        try:
            plan, plan_tokens = plan_calls(query, endpoints, use_llm, oas_paths_text)
            llm_calls = 0
            if use_llm:
                llm_calls += 2
            token_total = plan_tokens
            pass_rate = 0.0
            success_rate = 0.0
            executed_last: List[dict] = []
            if closed_loop and plan:
                tries = num_try
                for attempt in range(tries):
                    try:
                        executed = execute_calls(dataset, query, endpoints, plan)
                    except Exception as e:
                        error = str(e)
                        break
                    executed_last = executed
                    ok = sum(1 for e in executed if is_success(e.get('response')))
                    total = len(plan)
                    pass_rate = (ok / total) if total else 0.0
                    success_rate = pass_rate
                    if pass_rate >= 1.0:
                        break
                    if use_llm:
                        try:
                            from openai import OpenAI
                            api_key = os.getenv('OPENAI_API_KEY', '')
                            base_url = os.getenv('OPENAI_BASE_URL', '')
                            if not api_key or not base_url:
                                root = os.path.dirname(os.path.dirname(__file__))
                                cfg = load_simple_yaml(os.path.join(root, 'config.yaml'))
                                api_key = api_key or cfg.get('openai', '')
                                base_url = base_url or cfg.get('openai_base_url', '')
                            client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
                            err_txt = f'Executed {ok}/{total}.'
                            p_fix = build_fix_prompt(query, err_txt, endpoints, oas_paths_text)
                            r_fix = client.chat.completions.create(model='gpt-4o', messages=[{'role':'user','content':p_fix}], temperature=0)
                            t_fix = r_fix.choices[0].message.content
                            llm_calls += 1
                            token_total += total_tokens_of(r_fix)
                            revised = parse_calls(t_fix)
                            plan = revised if revised else plan
                        except Exception as e:
                            error = str(e)
                            break
            result_text = ''
            if dataset == 'spotify' and executed_last:
                result_text = responses_text(executed_last)
            elif use_llm and executed_last:
                try:
                    from openai import OpenAI
                    api_key = os.getenv('OPENAI_API_KEY', '')
                    base_url = os.getenv('OPENAI_BASE_URL', '')
                    if not api_key or not base_url:
                        root = os.path.dirname(os.path.dirname(__file__))
                        cfg = load_simple_yaml(os.path.join(root, 'config.yaml'))
                        api_key = api_key or cfg.get('openai', '')
                        base_url = base_url or cfg.get('openai_base_url', '')
                    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
                    p_ans = build_answer_prompt(query, executed_last)
                    r_ans = client.chat.completions.create(model='gpt-4o', messages=[{'role':'user','content':p_ans}], temperature=0)
                    result_text = r_ans.choices[0].message.content
                    llm_calls += 1
                    token_total += total_tokens_of(r_ans)
                except Exception as e:
                    error = str(e)
            t1 = time.time()
            cost_time = max(0.0, t1 - t0)
            rows.append([query, pass_rate, success_rate, llm_calls, token_total, result_text, error, cost_time])
        except Exception as e:
            t1 = time.time()
            cost_time = max(0.0, t1 - t0)
            rows.append([query, 0.0, 0.0, 0, 0, '', str(e), cost_time])
    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)

def _http_get(url: str, headers: dict, params: dict) -> dict:
    qs = urlencode(params or {})
    full = f"{url}?{qs}" if qs else url
    req = Request(full, headers=headers, method='GET')
    ctx = ssl.create_default_context()
    try:
        with urlopen(req, context=ctx) as resp:
            data = resp.read().decode('utf-8')
            try:
                return _json.loads(data)
            except Exception:
                return {'raw': data}
    except HTTPError as e:
        try:
            data = e.read().decode('utf-8')
        except Exception:
            data = ''
        return {'http_error_code': e.code, 'error': data or str(e)}
    except URLError as e:
        return {'url_error': str(e)}

def _http_request(method: str, url: str, headers: dict, params: dict, json_body: dict = None) -> dict:
    qs = urlencode(params or {})
    full = f"{url}?{qs}" if qs else url
    body_bytes = None
    hdrs = dict(headers or {})
    if json_body is not None:
        try:
            body_bytes = _json.dumps(json_body).encode('utf-8')
            hdrs['Content-Type'] = 'application/json'
        except Exception:
            body_bytes = None
    req = Request(full, headers=hdrs, method=method, data=body_bytes)
    ctx = ssl.create_default_context()
    try:
        with urlopen(req, context=ctx) as resp:
            data = resp.read().decode('utf-8')
            try:
                return _json.loads(data) if data else {'status': getattr(resp, 'status', 200)}
            except Exception:
                return {'raw': data}
    except HTTPError as e:
        try:
            data = e.read().decode('utf-8')
        except Exception:
            data = ''
        return {'http_error_code': e.code, 'error': data or str(e)}
    except URLError as e:
        return {'url_error': str(e)}

def plan_calls(query: str, endpoints: List[str], use_llm: bool, oas_text: str = '') -> Tuple[List[str], int]:
    if use_llm:
        try:
            from openai import OpenAI
            api_key = os.getenv('OPENAI_API_KEY', '')
            base_url = os.getenv('OPENAI_BASE_URL', '')
            if not api_key or not base_url:
                root = os.path.dirname(os.path.dirname(__file__))
                cfg = load_simple_yaml(os.path.join(root, 'config.yaml'))
                api_key = api_key or cfg.get('openai', '')
                base_url = base_url or cfg.get('openai_base_url', '')
            client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            p1 = build_solution_prompt(query, endpoints, oas_text)
            r1 = client.chat.completions.create(model='gpt-4o', messages=[{'role':'user','content':p1}], temperature=0)
            t1 = r1.choices[0].message.content
            tok = total_tokens_of(r1)
            plan_lines = parse_calls(t1)
            p2 = build_check_prompt("\n".join(plan_lines), endpoints, oas_text)
            r2 = client.chat.completions.create(model='gpt-4o', messages=[{'role':'user','content':p2}], temperature=0)
            t2 = r2.choices[0].message.content
            tok += total_tokens_of(r2)
            if '[Decision]: Yes' in t2:
                return plan_lines, tok
            revised = parse_calls(t2)
            return (revised if revised else plan_lines), tok
        except Exception:
            return rule_plan_calls(query, endpoints), 0
    return rule_plan_calls(query, endpoints), 0

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
    if 'collection' in q and has('/collection/{collection_id}/images'):
        calls.append('GET /collection/{collection_id}/images')
    if ('logo' in q or 'image' in q or 'poster' in q) and has('/company/{company_id}/images'):
        if has('/search/company'):
            calls.insert(0, 'GET /search/company')
        calls.append('GET /company/{company_id}/images')
    return calls if calls else (['GET /search/movie'] if has('/search/movie') else [])

def execute_calls(dataset: str, query: str, endpoints: List[str], calls: List[str]) -> List[dict]:
    out = []
    ctx = {}
    if dataset == 'tmdb':
        base = 'https://api.themoviedb.org/3'
        api_key = os.getenv('TMDB_API_KEY', '')
        if not api_key:
            root = os.path.dirname(os.path.dirname(__file__))
            cfg = load_simple_yaml(os.path.join(root, 'config.yaml'))
            api_key = cfg.get('tmdb', '')
        if not api_key:
            return out
        use_bearer = ('.' in api_key)
        headers = {'Authorization': f'Bearer {api_key}'} if use_bearer else {}
        for c in calls:
            m, p = c.split(' ', 1)
            if m != 'GET':
                continue
            params_base = {} if use_bearer else {'api_key': api_key}
            if p.startswith('/search/movie'):
                params = dict(params_base)
                params['query'] = extract_query_value(query, 'movie') or query
                res = _http_get(base + '/search/movie', headers, params)
                out.append({'call': c, 'response': res})
                try:
                    ctx['movie_id'] = res['results'][0]['id']
                except Exception:
                    pass
            elif p.startswith('/search/person'):
                params = dict(params_base)
                params['query'] = extract_query_value(query, 'person') or query
                res = _http_get(base + '/search/person', headers, params)
                out.append({'call': c, 'response': res})
                try:
                    ctx['person_id'] = res['results'][0]['id']
                except Exception:
                    pass
            elif p.startswith('/search/tv'):
                params = dict(params_base)
                params['query'] = extract_query_value(query, 'tv') or query
                res = _http_get(base + '/search/tv', headers, params)
                out.append({'call': c, 'response': res})
                try:
                    ctx['tv_id'] = res['results'][0]['id']
                except Exception:
                    pass
            elif p.startswith('/movie/top_rated'):
                params = dict(params_base)
                res = _http_get(base + '/movie/top_rated', headers, params)
                out.append({'call': c, 'response': res})
                try:
                    ctx['movie_id'] = res['results'][0]['id']
                except Exception:
                    pass
            elif p.startswith('/trending/'):
                parts = p.split('/')
                media_type = parts[2] if len(parts) > 2 else 'movie'
                time_window = parts[3] if len(parts) > 3 else 'day'
                params = dict(params_base)
                res = _http_get(base + f'/trending/{media_type}/{time_window}', headers, params)
                out.append({'call': c, 'response': res})
                try:
                    ctx['movie_id'] = res['results'][0]['id']
                except Exception:
                    pass
            elif p.startswith('/movie/') and 'credits' in p:
                mid = ctx.get('movie_id')
                if not mid:
                    continue
                params = dict(params_base)
                res = _http_get(base + f'/movie/{mid}/credits', headers, params)
                out.append({'call': c, 'response': res})
            elif p.startswith('/movie/') and 'keywords' in p:
                mid = ctx.get('movie_id')
                if not mid:
                    continue
                params = dict(params_base)
                res = _http_get(base + f'/movie/{mid}/keywords', headers, params)
                out.append({'call': c, 'response': res})
            elif p.startswith('/movie/') and 'images' in p:
                mid = ctx.get('movie_id')
                if not mid:
                    continue
                params = dict(params_base)
                res = _http_get(base + f'/movie/{mid}/images', headers, params)
                out.append({'call': c, 'response': res})
            elif p.startswith('/movie/') and 'reviews' in p:
                mid = ctx.get('movie_id')
                if not mid:
                    continue
                params = dict(params_base)
                res = _http_get(base + f'/movie/{mid}/reviews', headers, params)
                out.append({'call': c, 'response': res})
            elif p.startswith('/movie/') and 'similar' in p:
                mid = ctx.get('movie_id')
                if not mid:
                    continue
                params = dict(params_base)
                res = _http_get(base + f'/movie/{mid}/similar', headers, params)
                out.append({'call': c, 'response': res})
            elif p.startswith('/movie/') and 'recommendations' in p:
                mid = ctx.get('movie_id')
                if not mid:
                    continue
                params = dict(params_base)
                res = _http_get(base + f'/movie/{mid}/recommendations', headers, params)
                out.append({'call': c, 'response': res})
            elif p.startswith('/movie/') and 'release_dates' in p:
                mid = ctx.get('movie_id')
                if not mid:
                    continue
                params = dict(params_base)
                res = _http_get(base + f'/movie/{mid}/release_dates', headers, params)
                out.append({'call': c, 'response': res})
            elif p.startswith('/search/collection'):
                params = dict(params_base)
                params['query'] = extract_query_value(query, 'collection') or query
                res = _http_get(base + '/search/collection', headers, params)
                out.append({'call': c, 'response': res})
                try:
                    ctx['collection_id'] = res['results'][0]['id']
                except Exception:
                    pass
            elif p.startswith('/search/company'):
                params = dict(params_base)
                params['query'] = extract_query_value(query, 'company') or query
                res = _http_get(base + '/search/company', headers, params)
                out.append({'call': c, 'response': res})
                try:
                    ctx['company_id'] = res['results'][0]['id']
                except Exception:
                    pass
            elif p.startswith('/collection/') and 'images' in p:
                cid = ctx.get('collection_id')
                if not cid:
                    continue
                params = dict(params_base)
                res = _http_get(base + f'/collection/{cid}/images', headers, params)
                out.append({'call': c, 'response': res})
            elif p.startswith('/company/') and 'images' in p:
                coid = ctx.get('company_id')
                if not coid:
                    continue
                params = dict(params_base)
                res = _http_get(base + f'/company/{coid}/images', headers, params)
                out.append({'call': c, 'response': res})
            elif p.startswith('/movie/latest'):
                params = dict(params_base)
                res = _http_get(base + '/movie/latest', headers, params)
                out.append({'call': c, 'response': res})
                try:
                    ctx['movie_id'] = res.get('id')
                except Exception:
                    pass
            elif p.startswith('/person/') and 'movie_credits' in p:
                pid = ctx.get('person_id')
                if not pid:
                    continue
                params = dict(params_base)
                res = _http_get(base + f'/person/{pid}/movie_credits', headers, params)
                out.append({'call': c, 'response': res})
            elif p.startswith('/person/') and 'tv_credits' in p:
                pid = ctx.get('person_id')
                if not pid:
                    continue
                params = dict(params_base)
                res = _http_get(base + f'/person/{pid}/tv_credits', headers, params)
                out.append({'call': c, 'response': res})
            elif p.startswith('/tv/') and 'recommendations' in p:
                tid = ctx.get('tv_id')
                if not tid:
                    continue
                params = dict(params_base)
                res = _http_get(base + f'/tv/{tid}/recommendations', headers, params)
                out.append({'call': c, 'response': res})
            elif p.startswith('/tv/on_the_air'):
                params = dict(params_base)
                res = _http_get(base + '/tv/on_the_air', headers, params)
                out.append({'call': c, 'response': res})
            else:
                if '{' in p and '}' in p:
                    continue
                params = dict(params_base)
                res = _http_get(base + p, headers, params)
                out.append({'call': c, 'response': res})
    elif dataset == 'spotify':
        base = 'https://api.spotify.com/v1'
        token = os.getenv('SPOTIFY_TOKEN', '')
        if not token:
            root = os.path.dirname(os.path.dirname(__file__))
            cfg = load_simple_yaml(os.path.join(root, 'config.yaml'))
            token = cfg.get('spotipy_access_token', '') or cfg.get('spotify', '')
        if not token:
            return out
        headers = {'Authorization': f'Bearer {token}'}
        for idx, c in enumerate(calls):
            m, p = c.split(' ', 1)
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
                        ctx.setdefault('track_uris', []).append(it['uri'])
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
            elif p.startswith('/artists/') and '/top-tracks' in p:
                aid = ctx.get('artist_id')
                if not aid:
                    continue
                cnt = extract_number(query, 3)
                res = _http_get(base + f'/artists/{aid}/top-tracks', headers, {'market': 'US'})
                out.append({'call': c, 'response': res})
                try:
                    tracks = res.get('tracks', [])[:cnt]
                    uris = [t.get('uri') or f"spotify:track:{t.get('id')}" for t in tracks if t.get('id')]
                    ctx['track_uris'] = uris
                    if uris:
                        ctx['track_uri'] = uris[0]
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
                    ctx.setdefault('track_uris', []).append(ctx['track_uri'])
                except Exception:
                    pass
            elif p.startswith('/me'):
                res = _http_get(base + '/me', headers, {})
                out.append({'call': c, 'response': res})
                try:
                    ctx['user_id'] = res['id']
                except Exception:
                    pass
            elif p.startswith('/users/') and '/playlists' in p and m == 'POST':
                uid = ctx.get('user_id')
                if not uid:
                    me = _http_get(base + '/me', headers, {})
                    ctx['user_id'] = me.get('id')
                    uid = ctx['user_id']
                name = extract_query_value(query, 'playlist') or 'New Playlist'
                body = {'name': name, 'public': False}
                res = _http_request('POST', base + f'/users/{uid}/playlists', headers, {}, body)
                out.append({'call': c, 'request': {'body': body, 'params': {}}, 'response': res})
                try:
                    ctx['playlist_id'] = res.get('id')
                except Exception:
                    pass
            elif p.startswith('/playlists') and m == 'POST' and '{' not in p:
                uid = ctx.get('user_id')
                if not uid:
                    me = _http_get(base + '/me', headers, {})
                    ctx['user_id'] = me.get('id')
                    uid = ctx['user_id']
                name = extract_query_value(query, 'playlist') or 'New Playlist'
                body = {'name': name, 'public': False}
                res = _http_request('POST', base + f'/users/{uid}/playlists', headers, {}, body)
                out.append({'call': c, 'response': res})
                try:
                    ctx['playlist_id'] = res.get('id')
                except Exception:
                    pass
            elif p.startswith('/playlists/') and '/tracks' in p and m == 'POST':
                pid = ctx.get('playlist_id')
                uris = ctx.get('track_uris') or ([ctx.get('track_uri')] if ctx.get('track_uri') else [])
                if not pid or not uris:
                    continue
                body = {'uris': uris}
                res = _http_request('POST', base + f'/playlists/{pid}/tracks', headers, {}, body)
                out.append({'call': c, 'request': {'body': body, 'params': {}}, 'response': res})
            elif p.startswith('/me/player/queue') and m == 'POST':
                uri = ctx.get('track_uri')
                if not uri:
                    continue
                res = _http_request('POST', base + '/me/player/queue', headers, {'uri': uri}, None)
                out.append({'call': c, 'request': {'body': None, 'params': {'uri': uri}}, 'response': res})
            elif p.startswith('/me/player/next') and m == 'POST':
                res = _http_request('POST', base + '/me/player/next', headers, {}, None)
                out.append({'call': c, 'request': {'body': None, 'params': {}}, 'response': res})
            elif p.startswith('/me/player/volume') and m == 'PUT':
                vol = extract_number(query, 50)
                res = _http_request('PUT', base + '/me/player/volume', headers, {'volume_percent': vol}, None)
                out.append({'call': c, 'request': {'body': None, 'params': {'volume_percent': vol}}, 'response': res})
            elif p.startswith('/me/albums') and m == 'PUT':
                aid = ctx.get('album_id')
                if not aid:
                    continue
                res = _http_request('PUT', base + '/me/albums', headers, {'ids': aid}, None)
                out.append({'call': c, 'request': {'body': None, 'params': {'ids': aid}}, 'response': res})
            elif p.startswith('/me/following') and m == 'PUT':
                aid = ctx.get('artist_id')
                if not aid:
                    continue
                res = _http_request('PUT', base + '/me/following', headers, {'type': 'artist', 'ids': aid}, None)
                out.append({'call': c, 'request': {'body': None, 'params': {'type': 'artist', 'ids': aid}}, 'response': res})
            else:
                # Attempt to fill template ids
                filled = p
                for key in ['artist_id', 'album_id', 'track_id', 'playlist_id', 'user_id']:
                    placeholder = '{' + key + '}'
                    if placeholder in filled and key in ctx:
                        filled = filled.replace(placeholder, str(ctx[key]))
                if '{' in filled and '}' in filled:
                    continue
                if m == 'GET':
                    res = _http_get(base + filled, headers, {})
                    req_info = {'body': None, 'params': {}}
                elif m in ['POST', 'PUT', 'DELETE', 'PATCH']:
                    res = _http_request(m, base + filled, headers, {}, None)
                    req_info = {'body': None, 'params': {}}
                else:
                    res = {'unsupported_method': m, 'path': filled}
                    req_info = {'body': None, 'params': {}}
                out.append({'call': c, 'request': req_info, 'response': res})
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
            try:
                endpoints = list_endpoints({'paths': paths}) if paths else []
            except Exception:
                endpoints = []
            if not endpoints:
                paths_text = load_paths_text(oas_path)
                endpoints = list_endpoints_from_text(paths_text)
        if args.dataset == 'spotify':
            endpoints = spotify_supported_endpoints()
        oas_paths_text = load_paths_text(oas_path)
        data = []
        if args.query:
            data = [{'query': args.query}]
        else:
            data = load_json(ds_path)
            data = data[args.start:args.start+args.limit] if args.limit else data[args.start:]
        for item in data:
            query = item['query']
            calls, _ = plan_calls(query, endpoints, args.use_llm, oas_paths_text)
            resp = execute_calls(args.dataset, query, endpoints, calls)
            out_jsonl = os.path.join(base, 'experiment', 'result', f"{args.dataset}_responses.jsonl")
            ensure_dir(out_jsonl)
            with open(out_jsonl, 'a', encoding='utf-8') as jf:
                for r in resp:
                    jf.write(_json.dumps({'requirement': query, **r}, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()

