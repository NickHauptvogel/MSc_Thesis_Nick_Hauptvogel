set results_path=C:\Users\NHaup\OneDrive\Dokumente\Master_Studium\Semester_4\Thesis\Results

:: CNN LSTM IMDB
python .\util\ensemble_evaluation.py --path="%results_path%\imdb\original" -m=40 -cp=5
python .\util\ensemble_evaluation.py --path="%results_path%\imdb\bootstr" -m=40 -cp=1
python .\util\ensemble_evaluation.py --path="%results_path%\imdb\sse" -m=8 -cp=10
python .\util\ensemble_evaluation.py --path="%results_path%\imdb\epoch_budget" -m=20 -cp=1
:: ResNet20 CIFAR-10
python .\util\ensemble_evaluation.py --path="%results_path%\cifar10\resnet20\original" -m=24 -cp=5
python .\util\ensemble_evaluation.py --path="%results_path%\cifar10\resnet20\bootstr" -m=24 -cp=5
python .\util\ensemble_evaluation.py --path="%results_path%\cifar10\resnet20\sse" -m=24 -cp=5
python .\util\ensemble_evaluation.py --path="%results_path%\cifar10\resnet20\epoch_budget" -m=15 -cp=1
:: ResNet110 CIFAR-10
python .\util\ensemble_evaluation.py --path="%results_path%\cifar10\resnet110\original" -m=20 -cp=5
python .\util\ensemble_evaluation.py --path="%results_path%\cifar10\resnet110\bootstr" -m=20 -cp=5
python .\util\ensemble_evaluation.py --path="%results_path%\cifar10\resnet110\sse" -m=20 -cp=5
python .\util\ensemble_evaluation.py --path="%results_path%\cifar10\resnet110\epoch_budget" -m=15 -cp=1
:: WideResNet28x10 CIFAR-10
python .\util\ensemble_evaluation.py --path="%results_path%\cifar10\wideresnet2810\original" -m=24 -cp=5
python .\util\ensemble_evaluation.py --path="%results_path%\cifar10\wideresnet2810\bootstr" -m=24 -cp=5
python .\util\ensemble_evaluation.py --path="%results_path%\cifar10\wideresnet2810\sse" -m=24 -cp=5
python .\util\ensemble_evaluation.py --path="%results_path%\cifar10\wideresnet2810\epoch_budget" -m=15 -cp=1
:: ResNet110 CIFAR-100
python .\util\ensemble_evaluation.py --path="%results_path%\cifar100\resnet110\original" -m=24 -cp=5
python .\util\ensemble_evaluation.py --path="%results_path%\cifar100\resnet110\bootstr" -m=24 -cp=5
python .\util\ensemble_evaluation.py --path="%results_path%\cifar100\resnet110\sse" -m=24 -cp=5
python .\util\ensemble_evaluation.py --path="%results_path%\cifar100\resnet110\epoch_budget" -m=15 -cp=1
:: WideResNet28x10 CIFAR-100
python .\util\ensemble_evaluation.py --path="%results_path%\cifar100\wideresnet2810\original" -m=24 -cp=5
python .\util\ensemble_evaluation.py --path="%results_path%\cifar100\wideresnet2810\bootstr" -m=24 -cp=5
python .\util\ensemble_evaluation.py --path="%results_path%\cifar100\wideresnet2810\sse" -m=24 -cp=5
python .\util\ensemble_evaluation.py --path="%results_path%\cifar100\wideresnet2810\epoch_budget" -m=15 -cp=1
:: ResNet50 EyePACS
python .\util\ensemble_evaluation.py --path="%results_path%\retinopathy\resnet50\original" -m=8 -cp=6
python .\util\ensemble_evaluation.py --path="%results_path%\retinopathy\resnet50\bootstr" -m=8 -cp=6