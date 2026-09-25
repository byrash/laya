import json
import os
# import warnings
#
# warnings.filterwarnings(
#     "ignore",
#     message=r"laya: this checkpoint ships temperatures outside.*",
#     category=RuntimeWarning,
# )

os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
)

import torch,laya
from laya import Router

torch.set_num_threads(3)

device = "mps" if torch.backends.mps.is_available() else "cpu"

# model_path='./models/laya-multilingual'
model_path='./models/laya-english'
agent=laya.load(str(model_path),device=device)

questions={'department':{
 'type':'choice',
 'instructions':'Which team should handle this message?',
 'criteria':{
  'billing':'payments, duplicate charges, refunds',
  'technical':'software errors, bugs, login problems',
  'sales':'pricing or buying the product',
  'other':'unclear request or none of these',
 }}}
result=agent.system_one('I was charged twice. Please refund the duplicate payment.',questions)
print(json.dumps(result['answers']['department'],indent=2))


# Map the Router's "english" model to your local checkpoint.
router = Router(
    models={
        "english": model_path,
    },
    device=device,
    default="english",
    max_loaded=1,
)

state = (
    "Hi, we were billed twice for March. "
    "Please refund the duplicate today or we will cancel our plan."
)

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["not urgent", "soon", "blocking"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
}

result = router.predict(state, questions)

print(result["answers"]["department"]["choice"])
print(result["answers"]["churn_risk"]["noul"])
print(result["routing"]["model"])
