import json
import re
from pathlib import Path

readme = Path('README.md').read_text()
slides = readme.split('---')

template_html = Path('slides.template.html').read_text()

title_m = re.search(r'# (.*)', readme)
assert title_m is not None, 'Title not found in README.md'
title = title_m.group(1)

slides_html, count = re.subn('{{ *title *}}', title, template_html)
assert count == 1, f'Title found {count} times in slides.template.html'
slides_html, count = re.subn('{{ *slides *}}', lambda m: json.dumps(slides), slides_html)
assert count == 1, f'Slides found {count} times in slides.template.html'

Path('slides.html').write_text(slides_html)
