::python .\util\ensemble_prediction.py --path="C:\Users\NHaup\OneDrive\Dokumente\Master_Studium\Semester_4\Thesis\Results\imdb\original" -n=40 -cp=1
::python .\util\ensemble_prediction.py --path="C:\Users\NHaup\OneDrive\Dokumente\Master_Studium\Semester_4\Thesis\Results\imdb\bootstr" -n=40 -cp=1
::python .\util\ensemble_prediction.py --path="C:\Users\NHaup\OneDrive\Dokumente\Master_Studium\Semester_4\Thesis\Results\imdb\original_checkpointing" -n=8 -cp=5
::python .\util\ensemble_prediction.py --path="C:\Users\NHaup\OneDrive\Dokumente\Master_Studium\Semester_4\Thesis\Results\imdb\epoch_budget" -n=20 -cp=1
::python .\util\ensemble_prediction.py --path="C:\Users\NHaup\OneDrive\Dokumente\Master_Studium\Semester_4\Thesis\Results\imdb\sse" -n=8 -cp=10
::python .\util\ensemble_prediction.py --path="C:\Users\NHaup\OneDrive\Dokumente\Master_Studium\Semester_4\Thesis\Results\cifar10\resnet20\original" -n=24 -cp=5
::python .\util\ensemble_prediction.py --path="C:\Users\NHaup\OneDrive\Dokumente\Master_Studium\Semester_4\Thesis\Results\cifar10\resnet20\bootstr" -n=24 -cp=5
::python .\util\ensemble_prediction.py --path="C:\Users\NHaup\OneDrive\Dokumente\Master_Studium\Semester_4\Thesis\Results\cifar10\resnet20\sse" -n=24 -cp=5
::python .\util\ensemble_prediction.py --path="C:\Users\NHaup\OneDrive\Dokumente\Master_Studium\Semester_4\Thesis\Results\cifar10\resnet20\epoch_budget" -n=15 -cp=1

python .\util\ensemble_prediction.py --path="C:\Users\NHaup\Projects\Results\cifar100\wideresnet2810\original" -n=24 -cp=5
python .\util\ensemble_prediction.py --path="C:\Users\NHaup\Projects\Results\cifar100\wideresnet2810\bootstr" -n=24 -cp=5
python .\util\ensemble_prediction.py --path="C:\Users\NHaup\Projects\Results\cifar100\resnet110\sse" -n=24 -cp=5
python .\util\ensemble_prediction.py --path="C:\Users\NHaup\Projects\Results\cifar100\wideresnet2810\epoch_budget" -n=15 -cp=1

python .\util\ensemble_prediction.py --path="C:\Users\NHaup\Projects\Results\retinopathy\resnet50\original" -n=8 -cp=6
python .\util\ensemble_prediction.py --path="C:\Users\NHaup\Projects\Results\retinopathy\resnet50\bootstr" -n=8 -cp=6