set results_path=C:\Users\NHaup\OneDrive\Dokumente\Master_Studium\Semester_4\Thesis\Results
python .\util\plots.py --path="%results_path%\imdb" --m=40 --table
python .\util\plots.py --path="%results_path%\cifar10\resnet20" --m=40 --table
python .\util\plots.py --path="%results_path%\cifar10\resnet110" --m=20 --table
python .\util\plots.py --path="%results_path%\cifar10\wideresnet2810" --m=24 --table
python .\util\plots.py --path="%results_path%\cifar100\resnet110" --m=24 --table
python .\util\plots.py --path="%results_path%\cifar100\wideresnet2810" --m=24 --table
python .\util\plots.py --path="%results_path%\retinopathy\resnet50\" --m=8 --table