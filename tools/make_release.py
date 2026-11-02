"""Package the source data and analysis scripts into a versioned ZIP archive."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify import DATA_FILES, verify

FILES = {
    'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'CITATION.cff',
    '.gitignore', '.gitattributes', '.github/workflows/verify.yml', 'tools/make_release.py',
    'requirements.txt', 'reproduce.py', 'verify.py', 'parameters.json', 'plot_settings.json',
    'SHA256SUMS', *{'data/' + name for name in DATA_FILES},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dist')
    args = parser.parse_args()
    verify()
    contents = {}
    for name in sorted(FILES):
        path = ROOT/name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError('Unsafe source path: ' + name)
        contents[name] = path.read_bytes()
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    contents['SOURCE_MANIFEST.json'] = (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    citation = (ROOT/'CITATION.cff').read_text(encoding='utf-8')
    version = next(line.split(':', 1)[1].strip().strip('"') for line in citation.splitlines() if line.startswith('version:'))
    date = tuple(int(part) for part in version.split('.'))
    prefix = '2026-gpu-pt2-data-deposit-' + ''.join(f'{part:02d}' for part in date)
    archive = output/(prefix+'.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as stream:
        for name, data in sorted(contents.items()):
            item = zipfile.ZipInfo(prefix+'/'+name, (*date, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o644 << 16
            stream.writestr(item, data)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output/(prefix+'.sha256')).write_text(digest+'  '+archive.name+'\n', encoding='utf-8')
    print(f'Created {archive.name}: {len(FILES)} selected files, {archive.stat().st_size} bytes')


if __name__ == '__main__':
    main()
