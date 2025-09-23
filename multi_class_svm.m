%% RUN multi class SVM decoder with a timeline aligned to behavior onset 
%% load dataset 
clear
% load(path/to/E_Structure.mat)
%% configure decoder 
balance = true; % balance num of samples in each class
nshuf = 500; % number of shuffled control
window = [-10,10]; % timeline window (aligned to bv onset, in sec) to decode
labels = {'Huddle-Sniff-Rest','Active-Passive'};
bvlists = {[2,8,15],[3,4,5,6]};
%% run decoder for each animal 
accuracy = []; 
accuracy_sh = [];
fr = 15; 
baseline = [];
for a = 1:length(E)
    bv = E{a}.LogicalVecs;
    bvname = E{a}.EventNames;
    ca = E{a}.ms.FiltTraces(:,E{a}.ms.cell_label);
    ca = zscore(ca);
    % add a category with no behavior (if not exist) 
    if ~strcmp(bvname{1},'other')
        bv = [~any(bv,2),bv];
        bvname = ['other',bvname];
    end
    shift = randi([15*fr,size(ca,1)-15*fr],1,nshuf);
    for i = 1:length(labels)
        bvlist = bvlists{i};
        [accuracy{i}(a,:),nbouts{i}(a)] = prediction_timeline(ca,bv,bvlist,window,baseline,balance,fr);
        accsh = [];
        parfor s = 1:nshuf
            shifted = circshift(ca,shift(s),1);
            [accsh(s,:),~] = prediction_timeline(shifted,bv,bvlist,window,baseline,balance,fr) 
        end 
        accuracy_sh{i}(a,:) = mean(accsh); 
    end
end
%% plot decoder accuracy 
x = window(1):1/fr:window(2);
for i = 1:length(labels)
    keep = nbouts{i}>=0; % change this line if excluding animals with very little bouts 
    f = plot_accuracy_curve(accuracy{i},accuracy_sh{i},keep,x);
    xlabel('Time to bv(s)'); ylabel('Accuracy')
end
%%
function [X,Y,nbout,onsets_all] = make_bouts(ca,bv,bvlist,window,baseline,balance,fr)
% make bouts
X = []; Y = []; % x = time x neuron x trial 
onsets_all = [];
for i = 1:length(bvlist)
    CC = bwconncomp(bv(:,bvlist(i)));
    for b = 1:length(CC.PixelIdxList)
        onset= CC.PixelIdxList{b}(1);
        if onset+window(1)*fr + window(1)*fr <= 0 || onset+window(2)*fr > length(ca)
            continue % out of bound
        end
        if any(bv(onset+window(1)*fr:onset+window(2)*fr,11))
            continue % skip if any human intereference in the time window
        end
        onsets_all = [onsets_all,onset];
        if~isempty(baseline)
            if onset+baseline(1)*fr + baseline(1)*fr <= 0 || onset+baseline(2)*fr > length(ca)
                continue % out of bound
            end
            bl = mean(ca(onset+baseline(1)*fr:onset+baseline(2)*fr,:),1);
        else
            bl = 0;
        end
        len = min([window(2)*fr,length(CC.PixelIdxList{b})]);
        X = cat(3,X,ca(onset+window(1)*fr:onset+window(2)*fr,:)-bl);
        Y = [Y,i];
    end
end
nclass = length(unique(Y));
n_b = sum(Y==[1:nclass]',2);
nbout = min(n_b);
if balance
    sampled = [];
    for c = 1:nclass
        idx = find(Y == c);
        samp = randperm(sum(Y == c), nbout);
        sampled = [sampled,idx(samp)];
    end
    Y = Y(sampled); X = X(:,:,sampled); onsets_all = onsets_all(sampled);
end
end
function [X,Y,nbout] = balance_xy(X,Y)
nclass = length(unique(Y));
n_b = sum(Y==[1:nclass]',2);
nbout = min(n_b);
sampled = [];
for c = 1:nclass
    idx = find(Y == c);
    samp = randperm(sum(Y == c), nbout);
    sampled = [sampled,idx(samp)];
end
Y = Y(sampled); X = X(:,sampled); 
end
function [acrc,nbout] = prediction_timeline(ca,bv,bvlist,window,baseline,balance,fr)
% make bouts
[X_,Y,nbout,onsets_all] = make_bouts(ca,bv,bvlist,window,baseline,0,fr);
nt = size(X_,3);
for i = 1:size(X_,1)
    yhat = [];
    X = squeeze(X_(i,:,:));
    for k = 1:nt % leave one out cross validation
        test_idx = k;
        % exclude the bouts within 15 seconds
        excl = abs(onsets_all(k) - onsets_all) < 15*fr;
        train_idx = ~excl;
        
        Y_tr = Y(train_idx);X_tr = X(:,train_idx);
        % balance training 
        if balance
           [X_tr,Y_tr,nbout] = balance_xy(X_tr,Y_tr);
        end
        Mdl = fitcecoc(X_tr',Y_tr');
        yhat(test_idx) = predict(Mdl,X(:,test_idx)');
    end
acrc(i) = mean(yhat==Y);
end
end
function f = plot_accuracy_curve(acrc,shuf,mask,x)
    col = lines(2);
    f = figure;subplot(1,2,1); set(f,'PaperPosition',[0 0 7 4])
    plot(x,acrc(mask,:)'); box off; hold on
    line(xlim,[0.5 0.5],'Color','k','LineStyle','--')
    ylabel('Accuracy'); xlabel('Time to huddle (s)'); ylim([0 1]);xlim([-10 10])
    subplot(1,2,2)
    l(1) = line_ste_shade(acrc(mask,:),1,"x",x); hold on
    l(2) = line_ste_shade(shuf(mask,:),1,"x",x,"col",col(2,:));
    %line(xlim,[0.5 0.5],'Color','k','LineStyle','--')
    ylabel('Accuracy'); xlabel('Time to huddle (s)'); ylim([0 1]);xlim([-10 10])
    legend(l,{'Data','Shuffle'},'Box','off')
end