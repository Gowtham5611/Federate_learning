with open('README.md', 'r', encoding='utf-8') as f:
    text = f.read()

import re
text = re.sub(r'to \x07rchive/legacy_experiments/ with zero regressions', 'to `archive/legacy_experiments/` with zero regressions', text)
text = text.replace('**READY_WITH_WARNINGS**', '**`READY_WITH_WARNINGS`**')
text = text.replace('in phase7_readiness_audit/.', 'in `phase7_readiness_audit/`.')

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(text)

print('Done fixing readme')

