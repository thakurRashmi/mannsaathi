### Crisis-detection eval — `rules_only` mode

**140 cases · 0.0s · avg 0.01 ms/case**

| Metric | Value |
|---|---|
| Precision | **100.00%** |
| Recall    | **61.54%** |
| F1        | **76.19%** |
| Accuracy  | **82.14%** |
| False negatives (CRISES MISSED) | **25** |
| False positives (safe flagged)  | **0** |

**Per-category accuracy:**

| Category | Correct / Total | Accuracy |
|---|---|---|
| `crisis_direct` | 40/40 | 100.0% |
| `crisis_euphemism` | 0/25 | 0.0% |
| `safe_casual` | 15/15 | 100.0% |
| `safe_discussion` | 15/15 | 100.0% |
| `safe_distress` | 30/30 | 100.0% |
| `safe_figurative` | 15/15 | 100.0% |

**False negatives** (cases the system missed):
- `[crisis_euphemism]` "I'm tired of being here"
- `[crisis_euphemism]` 'I just want everything to stop'
- `[crisis_euphemism]` "I'm done with all of this. Permanently."
- `[crisis_euphemism]` "I don't want to wake up tomorrow"
- `[crisis_euphemism]` 'Nothing matters anymore. I just want it all to end.'
- `[crisis_euphemism]` 'I feel like everyone would be relieved if I just disappeared'
- `[crisis_euphemism]` "I've been thinking about how to make it stop forever"
- `[crisis_euphemism]` 'I just want to sleep and never wake up'
- `[crisis_euphemism]` "There's no point in being here"
- `[crisis_euphemism]` "I've made peace with not being around much longer"
- _… +15 more_
