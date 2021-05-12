# def decode_img(image_path):
#     img = tf.keras.preprocessing.image.load_img(
#         image_path, target_size=(shape))
#     img = tf.keras.preprocessing.image.smart_resize(img, shape)
#     img = tf.keras.preprocessing.image.img_to_array(
#         img)  # converted to ndarray
#     img = img.astype(np.float32)/255.0
#     img = np.expand_dims(img, axis=0)
#     return img


def save_users_images(mongo, image, username):
    mongo.save_file(image.filename, image)
    mongo.db.Diabetic.insert(
        {'username': username, 'diab_image': image.filename})
    return True
