"""Refresh the public file inventory after reviewing intentional release changes."""
import csv
import hashlib

from validate_release import R, public_files


def main():
    manifest = R / 'audit/release_manifest.csv'
    rows = []
    for path in public_files():
        if path == manifest:
            continue
        content = path.read_bytes()
        rows.append({
            'path': path.relative_to(R).as_posix(),
            'bytes': len(content),
            'sha256': hashlib.sha256(content).hexdigest(),
        })
    with manifest.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'bytes', 'sha256'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Updated {len(rows)} file records; run audit/validate_release.py next')


if __name__ == '__main__':
    main()
