function allbatchfeas=acalculate_layer_smap(im_data, segmentationData, rcnn_model,  p)
    imgsize = size(im_data);
    regionnum = size(segmentationData.inoutRectGt{p},1);
    boxes_1 = segmentationData.inoutRectGt{p}(:,1:4);
    boxes_2 = segmentationData.inoutRectGt{p}(:,5:8);
    boxes_1 = boxes_1(:, [3 1 4 2]);
    boxes_2 = boxes_2(:, [3 1 4 2]);
    correId = [repmat(p,[regionnum,1]), (1:regionnum)'];
    rcnnfeat_0 = rcnn_features_3(im_data, boxes_1, correId, segmentationData, rcnn_model);
    rcnnfeat_0 = rcnn_scale_features(rcnnfeat_0, rcnn_model.training_opts.feat_norm_mean);
    rcnnfeat_1 = rcnn_features1(im_data, boxes_1, correId, segmentationData, rcnn_model);
    rcnnfeat_1 = rcnn_scale_features(rcnnfeat_1, rcnn_model.training_opts.feat_norm_mean);
    rcnnfeat_2 = rcnn_features(im_data, boxes_2, rcnn_model);
    rcnnfeat_2 = rcnn_scale_features(rcnnfeat_2, rcnn_model.training_opts.feat_norm_mean);
    allbatchfeas = [rcnnfeat_1, rcnnfeat_2, rcnnfeat_0]';
end