rm results/*
cargo run --release -- exp1 data/instances_l50 --base-sigma=2 --num-sigma=25 -o results/exp1_s100_l50.csv -s=100
cargo run --release -- exp1 data/instances_l50 --base-sigma=2 --num-sigma=25 -o results/exp1_s200_l50.csv -s=200
cargo run --release -- exp1 data/instances_l50 --base-sigma=2 --num-sigma=25 -o results/exp1_s300_l50.csv -s=300

cargo run --release -- exp1 data/instances_l10 --base-sigma=2 --num-sigma=25 -o results/exp1_s100.csv -s=100
cargo run --release -- exp1 data/instances_l10 --base-sigma=2 --num-sigma=25 -o results/exp1_s150.csv -s=150
cargo run --release -- exp1 data/instances_l10 --base-sigma=2 --num-sigma=25 -o results/exp1_s200.csv -s=200

cargo run --release -- exp2 data/instances_l10 -o results/exp2_s100_l10.csv -s=100
cargo run --release -- exp2 data/instances_l50 -o results/exp2_s100_l50.csv -s=100 --num-preds=21

