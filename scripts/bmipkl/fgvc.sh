DEVICE=cuda:0

CFG=BMIPKL_fgvc
for SEED in 1 2 3
do
  bash scripts/bmipkl/base2new_train_bmipkl_fgvc.sh fgvc_aircraft ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl_fgvc.sh fgvc_aircraft ${SEED} ${DEVICE} ${CFG}
done

CFG=BMIPKL_fgvc1
for SEED in 1 2 3
do
  bash scripts/bmipkl/base2new_train_bmipkl_fgvc.sh fgvc_aircraft ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl_fgvc.sh fgvc_aircraft ${SEED} ${DEVICE} ${CFG}
done