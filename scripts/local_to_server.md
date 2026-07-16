로컬에서 코드를 작성한 뒤, 서버에서 그대로 실행할 때 필요한 최소값만 적어둔 메모입니다.
이 파일만 보고 셸 명령을 맞추면 됩니다.

```bash
cd /NHNHOME/WORKSPACE/0526050006_AA/wlaud1001/workspace/RoboMME_new
source scripts/activate_train_env.sh
export HF_HOME=/NHNHOME/WORKSPACE/0526050006_AA/wlaud1001/huggingface
hf auth login
ROBOVLA_GLOBAL_BATCH_SIZE=128 \
ROBOVLA_MAX_STEPS=100000 \
ROBOVLA_SAVE_STEPS=10000 \
scripts/train_robovla_robomme_full_ft.sh
```
