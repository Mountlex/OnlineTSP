rm results/*
cargo run --release -- exp1 data/instances_l50 --base-sigma=3 --num-sigma=20 -o results/exp1_s200_l50.csv -s=200
cargo run --release -- exp1 data/instances_l50 --base-sigma=3 --num-sigma=20 -o results/exp1_s350_l50.csv -s=350
cargo run --release -- exp1 data/instances_l50 --base-sigma=3 --num-sigma=20 -o results/exp1_s500_l50.csv -s=500

cargo run --release -- exp1 data/instances_l10 --base-sigma=3 --num-sigma=20 -o results/exp1_s100.csv -s=100
cargo run --release -- exp1 data/instances_l10 --base-sigma=3 --num-sigma=20 -o results/exp1_s150.csv -s=150
cargo run --release -- exp1 data/instances_l10 --base-sigma=3 --num-sigma=20 -o results/exp1_s200.csv -s=200

cargo run --release -- exp2 data/instances_l10 -o results/exp2_s100_l10.csv -s=100
cargo run --release -- exp2 data/instances_l50 -o results/exp2_s100_l50.csv -s=100 --num-preds=21

