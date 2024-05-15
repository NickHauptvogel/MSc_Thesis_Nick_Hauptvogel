:: CNN LSTM IMDB
set options=--performances --lr_loss --only_first --pac_bayes
set base_path=C:\Users\NHaup\OneDrive\Dokumente\Master_Studium\Semester_4\Thesis\Results\imdb
::python .\util\plots.py --path="%base_path%\original" --m=40 %options%
::python .\util\plots.py --path="%base_path%\original_checkpointing" --m=8 %options%
::python .\util\plots.py --path="%base_path%\bootstr" --m=40 %options%
::python .\util\plots.py --path="%base_path%\sse" --m=8 %options%
::python .\util\plots.py --path="%base_path%\epoch_budget" --m=20 %options%

:: ResNet20 CIFAR-10
set base_path=C:\Users\NHaup\OneDrive\Dokumente\Master_Studium\Semester_4\Thesis\Results\cifar10\resnet20
::python .\util\plots.py --path="%base_path%\original" --m=24 %options%
::python .\util\plots.py --path="%base_path%\bootstr" --m=24 %options%
::python .\util\plots.py --path="%base_path%\sse" --m=24 %options%
::python .\util\plots.py --path="%base_path%\epoch_budget" --m=15 %options%

:: ResNet110 CIFAR-10
set base_path=C:\Users\NHaup\Projects\Results\cifar10\resnet110
::python .\util\plots.py --path="%base_path%\original" --m=20 %options%
::python .\util\plots.py --path="%base_path%\bootstr" --m=20 %options%
::python .\util\plots.py --path="%base_path%\sse" --m=20 %options%
::python .\util\plots.py --path="%base_path%\epoch_budget" --m=15 %options%
::python .\util\plots.py --path="%base_path%\epoch_budget_300" --m=15 %options%

:: WideResNet28x10 CIFAR-10
set base_path=C:\Users\NHaup\Projects\Results\cifar10\wideresnet2810
::python .\util\plots.py --path="%base_path%\original" --m=24 %options%
::python .\util\plots.py --path="%base_path%\bootstr" --m=24 %options%
::python .\util\plots.py --path="%base_path%\sse" --m=24 %options%
::python .\util\plots.py --path="%base_path%\epoch_budget" --m=15 %options%
::python .\util\plots.py --path="%base_path%\epoch_budget_300" --m=15 %options%

:: ResNet110 CIFAR-100
set base_path=C:\Users\NHaup\Projects\Results\cifar100\resnet110
::python .\util\plots.py --path="%base_path%\original" --m=24 %options%
::python .\util\plots.py --path="%base_path%\bootstr" --m=24 %options%
::python .\util\plots.py --path="%base_path%\sse" --m=24 %options%
::python .\util\plots.py --path="%base_path%\epoch_budget" --m=15 %options%
::python .\util\plots.py --path="%base_path%\epoch_budget_300" --m=15 %options%

:: WideResNet28x10 CIFAR-100
set base_path=C:\Users\NHaup\Projects\Results\cifar100\wideresnet2810
::python .\util\plots.py --path="%base_path%\original" --m=24 %options%
::python .\util\plots.py --path="%base_path%\bootstr" --m=24 %options%
::python .\util\plots.py --path="%base_path%\sse" --m=24 %options%
::python .\util\plots.py --path="%base_path%\epoch_budget" --m=15 %options%
::python .\util\plots.py --path="%base_path%\epoch_budget_300" --m=15 %options%

:: ResNet50 EyePACS
set base_path=C:\Users\NHaup\Projects\Results\retinopathy\resnet50
::python .\util\plots.py --path="%base_path%\original" --m=8 %options%
python .\util\plots.py --path="%base_path%\bootstr" --m=8 %options%