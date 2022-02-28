rm results/*
cargo run --release -- exp1 data/instances_l10 --base-sigma=2 --num-sigma=25 -o results/exp1_s100.csv -s=100
cargo run --release -- exp1 data/instances_l10 --base-sigma=2 --num-sigma=25 -o results/exp1_s150.csv -s=150
cargo run --release -- exp1 data/instances_l10 --base-sigma=2 --num-sigma=25 -o results/exp1_s100_wc.csv -s=100 --wc-rd
cargo run --release -- exp1 data/instances_l10 --base-sigma=2 --num-sigma=25 -o results/exp1_s150_wc.csv -s=150 --wc-rd


cargo run --release -- exp2 data/instances_l10 -o results/exp2_s100.csv -s=100
cargo run --release -- exp2 data/instances_l10 -o results/exp2_s100_wc.csv -s=100 --wc-rd

