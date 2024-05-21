set results_path=PATH_TO_RESULTS

python .\util\plots.py --path="%results_path%\imdb" -m=40 --table
python .\util\plots.py --path="%results_path%\cifar10\resnet20" -m=24 --table
python .\util\plots.py --path="%results_path%\cifar10\resnet110" -m=20 --table
python .\util\plots.py --path="%results_path%\cifar10\wideresnet2810" -m=24 --table
python .\util\plots.py --path="%results_path%\cifar100\resnet110" -m=24 --table
python .\util\plots.py --path="%results_path%\cifar100\wideresnet2810" -m=24 --table
python .\util\plots.py --path="%results_path%\retinopathy\resnet50" -m=8 --table