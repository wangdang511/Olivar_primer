"""Inline docs/figs/*.png into walkthrough_template.html -> Olivar_walkthrough.html (self-contained)."""
import base64, re
from pathlib import Path
here = Path(__file__).parent
html = (here / 'walkthrough_template.html').read_text(encoding='utf-8')
def sub(m):
    data = base64.b64encode((here / 'figs' / f'{m.group(1)}.png').read_bytes()).decode()
    return f'data:image/png;base64,{data}'
html = re.sub(r'\{\{img:(\w+)\}\}', sub, html)
(here / 'Olivar_walkthrough.html').write_text(html, encoding='utf-8')
print('written', len(html)//1024, 'KB')
