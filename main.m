clear;
currentPath  = pwd;
addpath(genpath(currentPath ));
status=true;
reImgList=dir([pwd '/reimg/' '*.*g']);
if(length(reImgList)==0)
    fprintf('Please add the re-learned images or download our images from Google Cloud Drive.\n');
    status=false;
end

try
    run ./matconvnet/matlab/vl_setupnn
    net = load('./models/imagenet-vgg-f.mat') ;
    [rcnn_model, mdf] = init_deepmodel();
    init_para_mat = './models/seg_para.mat';
    load(init_para_mat, 'seg_para');
    nn1_model_name = './models/nn1.mat';
	load(nn_model_name, 'nn1');
    nn2_model_name = './models/nn2.mat';
	load(nn_model_name, 'nn2');
catch
    fprintf('Please download the model from Google Cloud Drive.\n');
    status=false;
end

if(status)
    img=imread([pwd '/img/00001.jpg']);
    imgDeepFeatures = single(img) ;
    imgDeepFeatures = imresize(imgDeepFeatures, net.meta.normalization.imageSize(1:2)) ;
    imgDeepFeatures = imgDeepFeatures - net.meta.normalization.averageImage ;
    
    res = vl_simplenn(net, imgDeepFeatures) ;
    
    featVec = res(20).x;
    
    featVec = featVec(:);
    deepFeatures = featVec';
    
    for reImgListId = 1:length(reImgList)
        try
            reimgDeepFeatures = imread([pwd '/reimg/' reImgList(reImgListId).name]);
            reimgDeepFeatures = single(reimgDeepFeatures) ;
            reimgDeepFeatures = imresize(reimgDeepFeatures, net.meta.normalization.imageSize(1:2)) ;
            reimgDeepFeatures = reimgDeepFeatures - net.meta.normalization.averageImage ;
            res = vl_simplenn(net, reimgDeepFeatures) ;
            
            featVec = res(20).x;
            
            featVec = featVec(:);
            deepFeatures = [deepFeatures; featVec'];
            fprintf('extract image name %s \n\n', reImgList(reImgListId).name);
            
        catch
            fprintf('+++++++++++++++error image name %s \n\n', reImgList(reImgListId).name);
        end
    end
    
    numRetrieval=10;
    numbbs=100;
    numsp=10;
    QueryVec = deepFeatures(1, :);
    [n,d] = size(deepFeatures);
    deepFeaturesScore = zeros(n, 1);
    
    deepFeaturesScore = (QueryVec*deepFeatures')';
    
    [~, deepFeaturesScoreId] = sort(deepFeaturesScore, 'descend');
    deepFeaturesScoreId=deepFeaturesScoreId-1;
    deepFeaturesScoreId=deepFeaturesScoreId(2:end);
    reimgDeepList={};
    for reImgDeepListId=1:numRetrieval
        reimgDeepList{reImgDeepListId,1}=reImgList(deepFeaturesScoreId(reImgDeepListId, 1)).name;
    end
    
    
    imageSize = [256,256];
    img= imresize(img,imageSize);
    imgHogFeatures = extractHOGFeatures(img,'CellSize',[4,4]);
    reimgHogFeatures=[];
    reimgHogDis=[];
    param.orientationsPerScale = [8 8 8 8];
    param.numberBlocks = 4;
    param.fc_prefilt = 4;
    [imgGistFeatures, ~] = LMgist(img, '', param);
    reimgGistDis=[];
    for reImgDeepListId=1:length(reimgDeepList)
        reimg=imread([pwd '/reimg/' char(reimgDeepList(reImgDeepListId))]);
        reimg= imresize(reimg,imageSize);
        reimgHogFeatures=extractHOGFeatures(reimg,'CellSize',[4,4]);
        reimgHogDis=[reimgHogDis;sum((imgHogFeatures-reimgHogFeatures).^2)];
        [reimgGistFeatures, ~] = LMgist(reimg, '', param);
        reimgGistDis=[reimgGistDis;sum((imgGistFeatures-reimgGistFeatures).^2)];
    end
    [~,hogRank]=sortrows(reimgHogDis);
    [~,gistRank]=sortrows(reimgGistDis);
    
    for reImgDeepListId=1:numRetrieval
        reimgDeepList{hogRank(reImgDeepListId,1),2}=reImgDeepListId;
        reimgDeepList{gistRank(reImgDeepListId,1),3}=reImgDeepListId;
    end
    matchImgList={};
    matchImgListId=1;
    for reImgDeepListId=1:numRetrieval
        if(double(reimgDeepList{reImgDeepListId,2})+double(reimgDeepList{reImgDeepListId,3})<numRetrieval)
            matchImgList{matchImgListId,1}=reimgDeepList{reImgDeepListId,1};
            matchImgListId=matchImgListId+1;
        end
    end
    
    
    model=load('edges/models/forest/modelBsds');
    model=model.model;
    model.opts.multiscale=0;
    model.opts.sharpen=2;
    model.opts.nThreads=4;
    opts = edgeBoxes;
    opts.alpha = .65;
    opts.beta  = .75;
    opts.minScore = .01;
    opts.maxBoxes = 1e4;
    imgbbs=edgeBoxes(img,model,opts);
    imgbbs=imgbbs(1:numbbs,:);
    matchImgProposalList={};
    matchImgProposalListId=1;
    tempmatchImgProposalList=[];
    for matchImgListId=1:length(matchImgList)
        tempreimg=imread([pwd '/reimg/' char(matchImgList{matchImgListId,1})]);
        tempreimg= imresize(tempreimg,imageSize);
        tempbbs=edgeBoxes(tempreimg,model,opts);
        tempbbs=sortrows(tempbbs,-5);
        matchImgList{matchImgListId,2}=  tempreimg;
        matchImgList{matchImgListId,3}=  tempbbs(1:numbbs,:);
        tempbbs=tempbbs(1:numbbs,:);
        
        im1=im2double(img);
        im2=im2double(tempreimg);
        
        cellsize=3;
        gridspacing=1;
        
        
        sift1 = mexDenseSIFT(im1,cellsize,gridspacing);
        sift2 = mexDenseSIFT(im2,cellsize,gridspacing);
        
        SIFTflowpara.alpha=2*255;
        SIFTflowpara.d=40*255;
        SIFTflowpara.gamma=0.005*255;
        SIFTflowpara.nlevels=4;
        SIFTflowpara.wsize=2;
        SIFTflowpara.topwsize=10;
        SIFTflowpara.nTopIterations = 60;
        SIFTflowpara.nIterations= 30;
        
        
        [vx,vy,energylist]=SIFTflowc2f(sift1,sift2,SIFTflowpara);
        
        matchImgList{matchImgListId,4}=  vx;
        matchImgList{matchImgListId,5}=  vy;
        
        [height2,width2,nchannels]=size(im2);
        [height1,width1]=size(vx);
        
        [xx,yy]=meshgrid(1:width2,1:height2);
        [XX,YY]=meshgrid(1:width1,1:height1);
        XX=XX+vx;
        YY=YY+vy;
        
        [temp1x, ~]=size(imgbbs);
        [temp2x, ~]=size(tempbbs);
        
        for temp1id=1:temp1x
            imgBoxes = [imgbbs(temp1id,1) imgbbs(temp1id,2) imgbbs(temp1id,1)+imgbbs(temp1id,3) imgbbs(temp1id,2)+imgbbs(temp1id,4)];
            imgRcnnfeat = rcnn_features(imgDeepFeatures, imgBoxes, rcnn_model);
            imgRcnnfeat = rcnn_scale_features(imgRcnnfeat, rcnn_model.training_opts.feat_norm_mean);
            for temp2id=1:temp2x
                tempXX=XX(imgbbs(temp1id,1):imgbbs(temp1id,2),imgbbs(temp1id,3):imgbbs(temp1id,4));
                tempYY=YY(imgbbs(temp1id,1):imgbbs(temp1id,2),imgbbs(temp1id,3):imgbbs(temp1id,4));
                tempxx=xx(tempbbs(temp2id,1):tempbbs(temp2id,2),tempbbs(temp2id,3):tempbbs(temp2id,4));
                tempyy=yy(tempbbs(temp2id,1):tempbbs(temp2id,2),tempbbs(temp2id,3):tempbbs(temp2id,4));
                tempXY=[tempXX(:) tempYY(:)];
                tempxy=[tempxx(:) tempyy(:)];
                [tempnum, ~ ,~]=intersect(tempxy,tempXY,'rows');
                [tempnumx,~]=size(tempnum);
                if(tempnumx>0)
                    matchImgProposalList{matchImgProposalListId,1}=img;
                    matchImgProposalList{matchImgProposalListId,2}=imgbbs(temp1id:temp1id,1:5);
                    matchImgProposalList{matchImgProposalListId,3}=tempreimg;
                    matchImgProposalList{matchImgProposalListId,4}=tempbbs(temp2id:temp2id,1:5);
                    matchImgProposalList{matchImgProposalListId,5}=tempnumx;
                    reBoxes = [tempbbs(temp2id,1) tempbbs(temp2id,2) tempbbs(temp2id,1)+tempbbs(temp2id,3) tempbbs(temp2id,2)+tempbbs(temp2id,4)];
                    reimgRcnnfeat = rcnn_features(tempreimg, reBoxes, rcnn_model);
                    reimgRcnnfeat = rcnn_scale_features(reimgRcnnfeat, rcnn_model.training_opts.feat_norm_mean);
                    matchImgProposalList{matchImgProposalListId,6}=imgRcnnfeat*reimgRcnnfeat';
                    tempmatchImgProposalList=[tempmatchImgProposalList;tempnumx,imgRcnnfeat*reimgRcnnfeat'];
                    matchImgProposalListId=matchImgProposalListId+1;
                end
            end
        end
    end
    
    [~,rankSiftFlow]=sortrows(tempmatchImgProposalList,-1);
    [~,rankRcnnfeat]=sortrows(tempmatchImgProposalList,-2);
    
    for tempmatchImgProposalListId=1:length(tempmatchImgProposalList)
        tempmatchImgProposalList(rankSiftFlow(tempmatchImgProposalListId,1),3)=tempmatchImgProposalListId;
        tempmatchImgProposalList(rankRcnnfeat(tempmatchImgProposalListId,1),4)=tempmatchImgProposalListId;
    end
    
    for tempmatchImgProposalListId=1:length(tempmatchImgProposalList)
        tempmatchImgProposalList(tempmatchImgProposalListId,5)=tempmatchImgProposalList(tempmatchImgProposalListId,3)+tempmatchImgProposalList(tempmatchImgProposalListId,4);
    end
    
    [~,rankTempmatchImgProposalList]=sortrows(tempmatchImgProposalList);
    
    matchImgSuperpixelList={};
    for matchImgSuperpixelListId=1:numsp
        matchImgSuperpixelList{matchImgSuperpixelListId,1}=matchImgProposalList{rankTempmatchImgProposalList(tempmatchImgProposalListId,1),1};
        matchImgSuperpixelList{matchImgSuperpixelListId,2}=matchImgProposalList{rankTempmatchImgProposalList(tempmatchImgProposalListId,1),2};
        matchImgSuperpixelList{matchImgSuperpixelListId,3}=matchImgProposalList{rankTempmatchImgProposalList(tempmatchImgProposalListId,1),3};
        matchImgSuperpixelList{matchImgSuperpixelListId,4}=matchImgProposalList{rankTempmatchImgProposalList(tempmatchImgProposalListId,1),4};
        matchImgSuperpixelList{matchImgSuperpixelListId,5}=matchImgProposalList{rankTempmatchImgProposalList(tempmatchImgProposalListId,1),5};
        matchImgSuperpixelList{matchImgSuperpixelListId,6}=matchImgProposalList{rankTempmatchImgProposalList(tempmatchImgProposalListId,1),6};
        tempimgProposalSuperpixel=getProposalSuperpixel(matchImgProposalList{rankTempmatchImgProposalList(tempmatchImgProposalListId,1),1},matchImgProposalList{rankTempmatchImgProposalList(tempmatchImgProposalListId,1),2},seg_para);
        matchImgSuperpixelList{matchImgSuperpixelListId,7}=tempimgProposalSuperpixel;
        tempreimgProposalSuperpixel=getProposalSuperpixel(matchImgProposalList{rankTempmatchImgProposalList(tempmatchImgProposalListId,1),3},matchImgProposalList{rankTempmatchImgProposalList(tempmatchImgProposalListId,1),4},seg_para);
        matchImgSuperpixelList{matchImgSuperpixelListId,8}= tempreimgProposalSuperpixel;
        for p=1:15
            imgallbatchfeas{p}=calculate_layer_smap(matchImgProposalList{rankTempmatchImgProposalList(tempmatchImgProposalListId,1),1}, tempimgProposalSuperpixel, rcnn_model, p);
            reimgallbatchfeas{p}=calculate_layer_smap(matchImgProposalList{rankTempmatchImgProposalList(tempmatchImgProposalListId,1),3}, tempreimgProposalSuperpixel, rcnn_model, p);
        end
        matchImgSuperpixelList{matchImgSuperpixelListId,9}=imgallbatchfeas;
        matchImgSuperpixelList{matchImgSuperpixelListId,10}=reimgallbatchfeas;
    end
    
    readyMatchImgSuperpixelList={};
    readyMatchImgSuperpixelListId=1;
    readyallbatchfeas=[];
    for matchImgSuperpixelListId=1:numsp
        tempimgallbatchfeas=matchImgSuperpixelList{matchImgSuperpixelListId,9};
        tempreimgallbatchfeas=matchImgSuperpixelList{matchImgSuperpixelListId,10};
        for p=1:15
            for tempimgallbatchfeasid=1:length(tempimgallbatchfeas)
                sctempimgallbatchfeas=tempimgallbatchfeas(tempimgallbatchfeasid,:)*tempreimgallbatchfeas';
                scvtempimgallbatchfeas=min(sctempimgallbatchfeas(sctempimgallbatchfeas>0));
                tempimgminallbatchfeas=find(sctempimgallbatchfeas==scvtempimgallbatchfeas);
                tempseg=matchImgSuperpixelList{matchImgSuperpixelListId,7};
                readyMatchImgSuperpixelList{readyMatchImgSuperpixelListId,1}.sp=tempseg{1,p};
                readyMatchImgSuperpixelList{readyMatchImgSuperpixelListId,1}.imgfeas=tempimgallbatchfeas(tempimgallbatchfeasid,:);
                readyMatchImgSuperpixelList{readyMatchImgSuperpixelListId,1}.refeas=tempreimgallbatchfeas(tempimgminallbatchfeas,:);
                readyMatchImgSuperpixelListId=readyMatchImgSuperpixelListId+1;
                readyallbatchfeas=[readyallbatchfeas;tempimgallbatchfeas(tempimgallbatchfeasid,:) tempreimgallbatchfeas(tempimgminallbatchfeas,:)];
            end
        end
        
    end
    [n1pro, n1labels] = nnpredict(nn1, readyallbatchfeas');
    n1labelsid=find(n1labels == 1);
    n1labelsList={};
    n1allbatchfeas=[];
    for n1labelsid_id=1:length(n1labelsid)
        n1labelsList{n1labelsid_id,1}=readyMatchImgSuperpixelList{n1labelsid(n1labelsid_id),1};
        n1labelsList{n1labelsid_id,2}=n1labelsid(n1labelsid_id);
        n1allbatchfeas=[n1allbatchfeas;readyMatchImgSuperpixelList{n1labelsid(n1labelsid_id),1}.imgfeas readyMatchImgSuperpixelList{n1labelsid(n1labelsid_id),1}.refeas];
    end
    
    [n2pro, n2labels] = nnpredict(nn2, n1allbatchfeas');
    
    n2pro(find(n2labels == 2)) = 1-n2pro(find(n2labels == 2));
    dssimg=imread([pwd '/img/00001_DSS.jpg']);
    
    for n1labelsid_id = 1:length(n1labelsid)
        allids = find(n1labelsList{n1labelsid_id,1}.sp.segimage == n1labelsList{n1labelsid_id,2});
        dssimg(allids) = n2pro(n1labelsid_id)*dssimg(allids);
    end
    
    imshow(dssimg);
end
