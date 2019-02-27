from metrics import *

def get_test_res(model, testGene, num_image):
    print("Test dataset contains %d" % num_image)
    dice_coef = 0.0
    jacc_coef = 0.0
    for i in range(num_image):
        (img, mask) = testGene.__next__()
        mask_pred = model.predict(img)
        mask_pred = np.reshape(mask_pred, (512, 512))
        # mask_pred[mask_pred >= 0.5] = 1
        # mask_pred[mask_pred < 0.5] = 0
        mask_true = np.reshape(mask, (512, 512))
        dice_coef += test_dice_coef(mask_true, mask_pred)
        tmp_jacc_coef = test_jacc_coef(mask_true, mask_pred)
        jacc_coef += tmp_jacc_coef if tmp_jacc_coef >= 0.65 else 0

        # io.imshow(mask_true)
        # io.show()
        # io.imshow(mask_pred)
        # io.show()
    print("In test dataset, dice_coef is %.5f and jacc_coef is %.5f" % (dice_coef / num_image, jacc_coef / num_image))
