% binary SVM decoder with a timeline aligned to behavior onset 
% inputs: 
% ca: neural activity (time x cell)
% bv: behavior matrix (time x behavior)
% bvlist: which two behaviors to decoder (2x1)
% window: decoding window in seconds, e.g. [-15 15]
% optional inputs:
% options.baseline: if given a window, the average activity over this window is subtracted as baseline
% options.balance: balance samples in classes 
% options.nrun: number of runs in data 
% options.nshift: number of runs in shuffle 

% outputs:
% acc: accuracy of prediction 
% shuf: accuracy of prediction in shuffle 
% auc: area under ROC curve 
% aucshuf: area under ROC curve in shuffle 
% nbout: number of bouts

function [acc,shuf,auc,aucshuf,nbout] = binary_svm_curve(ca,bv,bvlist,window,options)
arguments
    ca
    bv
    bvlist
    window
    options.baseline = []
    options.balance = true
    options.fr = 15
    options.alignatend = [0,0] % align to the end of behavior instead of the beginning 
    options.nrun = 50
    options.nshift = 500
end
    acc = []; shuf = []; nbout = [];
    shift = randi([15*fr,size(ca,1)-15*fr],1,options.nshift);
    parfor i = 1: options.nrun
        [acc(i,:),nbout(i),auc(i,:)] = prediction_curve(ca,bv,bvlist,window, options.baseline, options.balance, options.fr, options.alignatend);
    end
    parfor i = 1:length(shift)
        shifted = circshift(ca,shift(i),1);
        [shuf(i,:),~,aucshuf(i,:)] = prediction_curve(shifted,bv,bvlist,window, options.baseline, options.balance, options.fr, options.alignatend);
    end
    acc = mean(acc,1);
    auc = mean(auc,1);
    nbout = nbout(1);
    shuf = mean(shuf,1);
    aucshuf = mean(aucshuf,1);
end

function [acrc,nbout,auc] = prediction_curve(ca,bv,bvlist,window,baseline,balance,fr,alignatend)
% make bouts
X = []; Y = []; % x = time x neuron x trial
onsets_all = [];
for i = 1:length(bvlist)
    CC = bwconncomp(bv(:,bvlist(i)));
    for b = 1:length(CC.PixelIdxList)
        
        if alignatend(i)
            onset= CC.PixelIdxList{b}(end)+1;
        else
            onset= CC.PixelIdxList{b}(1);
        end
        if onset+window(1)*fr  <= 0 || onset+window(2)*fr > length(ca)
            continue % out of bound
        end
        if any(bv(onset+window(1)*fr:onset+window(2)*fr,11))
            continue % skip if any human interference in the time window
        end

        onsets_all = [onsets_all,onset]; 
        % take 10s activity surrounding huddle onset
        if ~isempty(baseline)
            bl = mean(ca(onset+baseline(1)*fr:onset+baseline(2)*fr,:),1);
        else
            bl = 0;
        end
        X = cat(3,X,ca(onset+window(1)*fr:onset+window(2)*fr,:)-bl);
        Y = [Y,i];

    end
end

% balance the two classes
n_b = [sum(Y==1),sum(Y==2)];
nbout = min(n_b);
if nbout <= 3 
    %disp('less than 3 bouts')
    acrc=nan;auc=nan;
else
if balance
    if n_b(1)<n_b(2)
        shuf = randperm(n_b(2));
        x = (n_b(1)+1):sum(n_b);
        take = [1:n_b(1),x(shuf(1:nbout))];
        X = X(:,:,take);
        Y = Y(take);
        onsets_all = onsets_all(take);
    else
        shuf = randperm(n_b(1));
        x = 1:n_b(1);
        take = [x(shuf(1:nbout)),n_b(1)+1:sum(n_b)];
        X = X(:,:,take);
        Y = Y(take);
        onsets_all = onsets_all(take);
    end
end

yhat = [];
for t = 1:size(X,1)
    xx = squeeze(X(t,:,:))';
    nt = size(X,3);
    for k = 1:nt % leave one out cross validation
        test_idx = k;
        % exclude the bouts within 15 seconds
        excl = abs(onsets_all(k) - onsets_all) < 15*fr;
        train_idx = ~excl;
        %disp(sum(train_idx))
        Y_tr = Y(train_idx)';X_tr = xx(train_idx,:)';
        % balance training 
        if sum(Y_tr==Y(test_idx))<sum(Y_tr~=Y(test_idx))
        rmv = randsample(find(Y_tr~=Y(test_idx)),1);
        X_tr(:,rmv) = []; Y_tr(rmv) = [];
        end
        SVMModel = fitcsvm(X_tr',Y_tr');
        [yhat(test_idx),score(test_idx,:)] = predict(SVMModel,xx(test_idx,:));
    end
    acrc(t) = mean(yhat==Y);
    [~,~,~,auc(t)] = perfcurve(Y,score(:,1),1);
end
end
end
