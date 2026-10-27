"""Download bread image candidates for human review (not a labeled final dataset)."""
from pathlib import Path
from io import BytesIO
import argparse
import hashlib
import random
import time

import requests
from PIL import Image, ImageOps, UnidentifiedImageError
from ddgs import DDGS

BASE = Path(__file__).resolve().parent
QUERIES = {
    'fresh': [
    'white sandwich bread slices photography',
    'plain fresh white bread slice closeup',
    'sliced white bread loaf on table',
    'fresh bread slices isolated white background',
    'soft white sandwich bread close up',
    'fresh whole wheat bread slices',
    'plain bread loaf sliced food photography',
    'fresh toast bread slices before toasting',
    'white sandwich bread stack closeup',
    'fresh brown bread slices photography',
    'bakery fresh bread loaf cut slices',
    'unspoiled sandwich bread slice',
],
    'spoiled': [
        'moldy bread slice green mold close up -person -people -meme -cartoon',
        'spoiled bread mold spots close up photo -person -people -meme -cartoon',
        'bread fungal mold growth slice photo -person -people -meme -cartoon',
    ],
}
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36',
           'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8'}
MAX_BYTES = 15 * 1024 * 1024

def download(session, url):
    try:
        with session.get(url, headers=HEADERS, timeout=(8, 18), stream=True, allow_redirects=True) as response:
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_content(chunk_size=65536):
                content.extend(chunk)
                if len(content) > MAX_BYTES:
                    return None
        with Image.open(BytesIO(content)) as im:
            im.load()
            if getattr(im, 'is_animated', False):
                im.seek(0)
            im = ImageOps.exif_transpose(im).convert('RGB')
            if min(im.size) < 160:
                return None
            return im
    except (requests.RequestException, UnidentifiedImageError, OSError, ValueError):
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=150, help='candidate images per class')
    parser.add_argument('--max-results', type=int, default=250, help='search results per query')
    parser.add_argument('--class-name', choices=['fresh', 'spoiled', 'both'], default='both')
    args = parser.parse_args()
    if args.target < 1 or args.max_results < 1:
        parser.error('target and max-results must be positive')
    root = BASE / 'review_candidates'
    session = requests.Session()
    for label, queries in QUERIES.items():
        if args.class_name != 'both' and label != args.class_name:
            continue

        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        existing = list(folder.glob('*.jpg'))
        seen = set()
        for p in existing:
            try:
                with Image.open(p) as im:
                    seen.add(hashlib.sha256(im.convert('RGB').resize((64,64)).tobytes()).hexdigest())
            except OSError:
                pass
        count = len(existing)
        print(f'\n[{label}] Existing: {count} | Target: {args.target}', flush=True)
        if count >= args.target:
            continue
        urls = []
        for query in queries:
            try:
                results = DDGS().images(query, max_results=args.max_results, safesearch='on')
                urls.extend([item.get('image') for item in results if item.get('image')])
                print(f'Found {len(urls)} candidate URLs so far', flush=True)
            except Exception as exc:
                print(f'Search error (continuing): {exc}', flush=True)
            time.sleep(1)
        random.shuffle(urls)
        visited = set()
        for index, url in enumerate(urls, 1):
            if count >= args.target:
                break
            if url in visited:
                continue
            visited.add(url)
            im = download(session, url)
            if im is None:
                continue
            signature = hashlib.sha256(im.resize((64,64)).tobytes()).hexdigest()
            if signature in seen:
                continue
            seen.add(signature)
            filename = folder / f'{label}_{count+1:04d}.jpg'
            try:
                im.save(filename, 'JPEG', quality=90, optimize=True)
                count += 1
                if count % 10 == 0:
                    print(f'[{label}] Saved {count}/{args.target} (checked {index} URLs)', flush=True)
            except OSError as exc:
                print(f'Save error: {exc}', flush=True)
        print(f'[{label}] Finished with {count} images. Review manually before using.', flush=True)
    print('\nDone. Review candidates and copy only correct images into dataset_raw/fresh or dataset_raw/spoiled.')

if __name__ == '__main__':
    main()
