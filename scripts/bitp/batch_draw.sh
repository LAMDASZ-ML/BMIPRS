TRAINER=$1
CFG=$2
SUB=$3
for SEED in 1 2 3
do
  bash scripts/bitp/base2new_draw_bitp.sh caltech101 ${SEED} ${TRAINER} ${CFG} ${SUB}
  bash scripts/bitp/base2new_draw_bitp.sh dtd ${SEED} ${TRAINER} ${CFG} ${SUB}
  bash scripts/bitp/base2new_draw_bitp.sh eurosat ${SEED} ${TRAINER} ${CFG} ${SUB}
  bash scripts/bitp/base2new_draw_bitp.sh fgvc_aircraft ${SEED} ${TRAINER} ${CFG} ${SUB}
  bash scripts/bitp/base2new_draw_bitp.sh food101 ${SEED} ${TRAINER} ${CFG} ${SUB}
  bash scripts/bitp/base2new_draw_bitp.sh oxford_flowers ${SEED} ${TRAINER} ${CFG} ${SUB}
  bash scripts/bitp/base2new_draw_bitp.sh oxford_pets ${SEED} ${TRAINER} ${CFG} ${SUB}
  bash scripts/bitp/base2new_draw_bitp.sh stanford_cars ${SEED} ${TRAINER} ${CFG} ${SUB}
  bash scripts/bitp/base2new_draw_bitp.sh ucf101 ${SEED} ${TRAINER} ${CFG} ${SUB}
done
exit 0

  bash scripts/bitp/base2new_draw_bitp.sh sun397 ${SEED} ${TRAINER} ${CFG} ${SUB}
  bash scripts/bitp/base2new_draw_bitp.sh imagenet ${SEED} ${TRAINER} ${CFG} ${SUB}
