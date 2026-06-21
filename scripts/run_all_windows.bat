@echo off
REM Run the full battery locally on Windows: balanced + leakage-free, all 5 classical
REM models, then a cluster permutation test per classifier. Results go to local_results\.
REM Needs zuco_features.npz and zuco_montage.npz in the repo root, and a .venv created with
REM   uv venv --python 3.12  &&  uv pip install mne scikit-learn pandas matplotlib
REM Run from anywhere (it cd's to the repo root):  scripts\run_all_windows.bat

cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe
set BASE=scripts\run_local.py --features zuco_features.npz --montage zuco_montage.npz --grouped --scoring balanced --no-plot --n-jobs -1

echo ===== classical searchlight (all 5 models, balanced, grouped) =====
%PY% %BASE%

echo ===== permutation: LDA (1000) =====
%PY% %BASE% --permutation --clf lda --skip-classical --n-perm 1000

echo ===== permutation: LogReg (1000) =====
%PY% %BASE% --permutation --clf logreg --skip-classical --n-perm 1000

echo ===== permutation: RandomForest (5 x 100, checkpointed) =====
for /L %%i in (1,1,5) do %PY% %BASE% --permutation --clf rf --skip-classical --n-perm 100

echo ===== permutation: MLP (5 x 100, checkpointed) =====
for /L %%i in (1,1,5) do %PY% %BASE% --permutation --clf mlp --skip-classical --n-perm 100

echo ===== permutation: SVM (5 x 100, checkpointed, slowest) =====
for /L %%i in (1,1,5) do %PY% %BASE% --permutation --clf svm --skip-classical --n-perm 100

echo ===== ALL DONE - results are in local_results\ =====
pause
