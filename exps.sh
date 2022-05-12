rm results/*

cargo run --release -- exp1 data/instances --base-sigma=2 --num-sigma=25 -o results/exp1_s100.csv -s=100
cargo run --release -- exp1 data/instances --base-sigma=2 --num-sigma=25 -o results/exp1_s100_rd.csv -s=100 --correct-rd
cargo run --release -- exp2 data/instances -o results/exp2_s100_l10.csv -s=100

