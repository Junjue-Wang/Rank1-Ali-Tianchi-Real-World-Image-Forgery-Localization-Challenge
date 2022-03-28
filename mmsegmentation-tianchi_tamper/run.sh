./tools/dist_test.sh \
    work_dirs/binzhou/convx_l_2x_ft_all/binzhou_convx_l_ft.py \
    work_dirs/binzhou/convx_l_2x_ft_all/epoch_24.pth \
    4 \
    --eval-options imgfile_prefix="./work_imgs" \
    --format-only
mkdir results
python -c "import os; import cv2; from skimage import io; [io.imsave(os.path.join('./results', x.replace('GF', 'LT').replace('png', 'tif')), cv2.imread('./work_imgs/' + x, cv2.IMREAD_UNCHANGED) + 1) for x in os.listdir('./work_imgs')]" > /dev/null
cd results; zip results.zip -r -9 *.tif; mv results.zip ../data/binzhou/; cd ..; rm results -rf

./tools/dist_test.sh \
    work_dirs/tamper/convx_b_team_30k/tamper_convx_b_team.py \
    work_dirs/tamper/convx_b_team_30k/latest.pth \
    4 \
    --options \
    model.test_cfg.logits=True \
    model.test_cfg.binary_thres=0.5 \
    data.test.pipeline.1.flip=True \
    data.test.pipeline.1.img_ratios="[2.0]" \
    data.test.pipeline.1.flip_direction="['horizontal']" \
    --format-only \
    --eval-options imgfile_prefix="./data/tamper/test/logits" \
    data.test.img_dir="train2/img" \
    data.test.ann_dir="train2/msk" \
    --eval mIoU mFscore \

python -c "import glob, cv2; [cv2.imwrite(_, cv2.imread(_, cv2.IMREAD_UNCHANGED) * 255) for _ in glob.glob('./data/tamper/test/images/*')]"; cd ./data/tamper/test; zip images.zip -9 -r ./images

python -c "import glob, cv2; print(sum([(cv2.imread(_, cv2.IMREAD_UNCHANGED) == 0).all() for _ in glob.glob('./data/tamper/test/images/*')]))"