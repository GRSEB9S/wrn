from keras import layers, models, regularizers


def building_block(inputs, double_filters, weight_decay, dropout):
    """Create basic building block for WRN."""
    try:
        layer_id = building_block.id
    except AttributeError:
        building_block.id = 1
        layer_id = 1
    x = layers.BatchNormalization(name='l' + str(layer_id) + '_bn1')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)
    if double_filters:
        n_filters = int(inputs.shape[-1]) * 2
        add = layers.Conv2D(n_filters, (2, 2), kernel_regularizer=regularizers.l2(weight_decay), strides=(2, 2), trainable=False, name='l' + str(layer_id) + '_Ws')(inputs)
        stride = 2
        x = layers.Conv2D(n_filters, (3, 3), kernel_regularizer=regularizers.l2(weight_decay), strides=(stride, stride), padding='same', name='l' + str(layer_id) + '_conv1_3x3x' + str(n_filters) + '_stride_' + str(stride))(x)
    else:
        n_filters = int(inputs.shape[-1])
        add = inputs
        x = layers.Conv2D(n_filters, (3, 3), kernel_regularizer=regularizers.l2(weight_decay), padding='same', name='l' + str(layer_id) + '_conv1_3x3x' + str(n_filters))(x)

    x = layers.BatchNormalization(name='l' + str(layer_id) + '_bn2')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(n_filters, (3, 3), kernel_regularizer=regularizers.l2(weight_decay), padding='same', name='l' + str(layer_id) + '_conv2_3x3x' + str(n_filters))(x)
    building_block.id += 1
    return layers.Add(name='l' + str(layer_id) + '_add')([x, add])


def build_wrn(inputs, n_classes, first_layer_kernel=(3, 3), first_layer_strides=(1, 1), groups=3, blocks_in_groups=1, filters_mult=1, dropout=0., weight_decay=0., include_softmax=True):
    """Create WRN."""
    n_filters_1st_layer = 16 * filters_mult
    t = layers.BatchNormalization(name='bn0')(inputs)
    t = layers.Conv2D(n_filters_1st_layer, first_layer_kernel, kernel_regularizer=regularizers.l2(weight_decay), strides=first_layer_strides, padding='same', activation='relu', name='conv0_7x7x' + str(n_filters_1st_layer) + '_stride2')(t)
    for i in range(groups):
        t = building_block(t, i > 0, weight_decay, dropout)
        for _ in range(blocks_in_groups - 1):
            t = building_block(t, False, weight_decay, dropout)
    t = layers.BatchNormalization(name='last_bn')(t)
    t = layers.Activation('relu')(t)
    t = layers.pooling.GlobalAveragePooling2D(name='global_avg_pool')(t)
    t = layers.Dense(n_classes, kernel_regularizer=regularizers.l2(weight_decay), name='output')(t)
    if include_softmax:
        t = layers.Activation('softmax')(t)
    return models.Model(inputs=inputs, outputs=t)
