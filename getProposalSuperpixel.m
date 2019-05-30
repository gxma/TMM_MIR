function segmentationData = getProposalSuperpixel( image,bbs,seg_para )
segmentationData = [];
num_segmentation = 15;
segType    = 'pedro';

for p = 1 : num_segmentation
    segmentationData.segmat{p} = im2superpixels( image, segType, seg_para(p, :) );
    for k = 1:segmentationData.segmat{p}.nseg
        [rc] = find(segmentationData.segmat{p}.segimage == k);
        
        [r1, c1] = find(segmentationData.segmat{p}.segimage == k);
        if(bbs(1)<min(c1) & bbs(2)<min(r1) & bbs(1)+bbs(3)>max(c1) & bbs(2)+bbs(4)>max(r1))
            rect = [min(r1), max(r1), min(c1), max(c1)];
            adjids = find(segmentationData.segmat{p}.adjmat(k,:) == 1);
            r = [];
            c = [];
            for m = 1:size(adjids, 2)
                [tmpr, tmpc] = find(segmentationData.segmat{p}.segimage == adjids(m));
                r = [r;tmpr];
                c = [c;tmpc];
            end
            rect = [rect, min(r), max(r), min(c),max(c)];
            segmentationData.inoutRectGt{p}(k,:) = rect;
        end
    end
end

end

