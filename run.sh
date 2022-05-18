max=3
for i in `seq 0 $max`
do
  nohup python main.py $i > main$i.out 2>&1 &
done