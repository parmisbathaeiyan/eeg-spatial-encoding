@echo off
REM Run the full TeCo battery locally on Windows: balanced + leakage-free, all 5 classical
REM models, then a cluster permutation test per classifier. Results go to teco_results\.
REM Needs teco_features.npz and teco_montage.npz in the repo root, and a .venv created with
REM   uv venv --python 3.12  &&  uv pip install mne scikit-learn pandas matplotlib h5py
REM Run from anywhere (it cd's to the repo root):  scripts\run_all_teco_windows.bat

cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe
set BASE=scripts\run_local.py --features teco_features.npz --montage teco_montage.npz --montage-rotate 0 --grouped --scoring balanced --no-plot --n-jobs -1 --out-dir teco_results

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

echo ===== ALL DONE - results are in teco_results\ =====
pause
