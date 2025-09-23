function rocall = roc_all(ca,bv,options)
% Run receiver oparating characteristic (ROC) analysis on each behavior, 
% circularly shift 1000 times to generate a null distribution and test
% significance of each cell. 

% Inputs:
%   neural population activity (cell x time)
%   behavioral matrix (behavior x time) by default the first column
%   represents the negative class (when no behavior happened)

% Optional inputs:   
arguments
    ca % calium trace (cell x time)
    bv % behavior (behavior x time)
    options.define_negative_class logical = 1 % 0 -> take the entire session, 1 -> compare each behavior to a specific culumn for negative class 
    options.negative = 1 % which column is the negative class
    options.nshift = 1000 % number of shuffle
    options.bvlist = [] % the list of behaviors to analyse 
    options.save logical = 1 
    options.path = []
    options.rocname = [] % name of the behaviors in bvmat 
end

% Example usage:
%    rocall = roc_all(ca,rocmat,'bvlist',[2,3,4],'negative',1,'path',save_path_dir,'rocname',{'correct','miss','opaque'});

% Output:
% a structure named rocall with the following fields:
%   rocmat: behavior matrix (behavior x time)
%   rocname: name of each column of the behavior matrix 
%   auc: area under ROC curve (behavior x cell)
%   aucrand: area under ROC curve in the null distribution (behavior x cell
%   x nshift)
%   pvalue: pvalue of each cell (behavior x cell)
%   nsig: number of significantly activated/suppressed neurons (behavior x 2)

% a significantly activated neuron is a neuron with auc > 0.5 and 
% p-value < 0.05, a significantly suppressed neuron is a neuron with auc <0.5 and
% p-value < 0.05

% rocall is saved to the saving path defined in inputs

% generate negative class idx 
if options.define_negative_class
    nullind=bv(options.negative,:); 
else 
    nullind=true(1,size(bv,2));
end
% generate behavioral list 
if isempty(options.bvlist) % run all columns
    bvlist=1:size(bv,1);
    bvlist = setdiff(bvlist,options.negative); % all behavior except for other
else
    bvlist = options.bvlist; 
end
% make sure no bv column is empty 
emp = sum(bv(bvlist,:),2) ==0;
if sum(emp)~=0
    warning('behavior %s is empty',num2str(find(emp)))
    % skip runing for empty behaviors 
    run_idx = find(~emp);
else
    run_idx = 1:length(bvlist);
end
n_bv = length(bvlist); ncell = size(ca,1);
% run roc 
rocall.nsig = nan(n_bv,2);
rocall.auc = nan(n_bv,ncell);
rocall.aucrand = nan(n_bv,ncell,options.nshift);
for i = 1:length(run_idx)
    b = run_idx(i);
    ind = any([nullind;bv(bvlist(b),:)]);
    [~,~,rocall.auc(b,:)] = roc(ca(:,ind),bv(bvlist(b),ind));
    rocall.aucrand(b,:,:) = aucshift(ca(:,ind),bv(bvlist(b),ind),options.nshift);
end

% get p value
[~,~,rocall.pvalue,rocall.nsig,rocall.overlap_act,rocall.overlap_sup]=rocstats(rocall.aucrand,rocall.auc);
rocall.bvlist = bvlist;
rocall.rocmat = bv;
if ~isempty(options.rocname)
    rocall.rocname = options.rocname;
end

% save
if options.save&&isempty(options.path)
    [filepath,~,~] = fileparts(options.path);
    mkdir(filepath)
    save('roc.mat','rocall','-v7.3')
elseif options.save&&~isempty(options.path)
    save(options.path,'rocall','-v7.3')
end

function [FPR,TPR,auc]=roc(sig,beh)
% inputs: sig = signal in (cell,time), beh = behavior in (behavior, time)
[nc,~]=size(sig);nb=size(beh,1);
auc=NaN(nb,nc);TPR=cell(nb,nc);FPR=cell(nb,nc);
for b = 1:nb
    for i = 1:nc
        if all(sig(i,:)==sig(i,1))
            sprintf('neuron%d,signal empty',i)
        end
        [FPR{b,i},TPR{b,i},~,auc(b,i)] = perfcurve(beh(b,:),sig(i,:),1);
    end
end

function aucrand=aucshift(signal,label,nshift,options)
% roc analysis for random shift signals
% inputs: signal = 2D matrix (cell, times), label = 2D matrix (behavior,
% times), nshift = number of shifts
% outputs: aucrand = 3D matrix (labels,cells,shifts)
arguments
    signal
    label
    nshift
    options.limit = [] % limit of shuffle range, input specific number as number of time bins
end
[nc,len]=size(signal);
if isempty(options.limit)
    shift = randi([1,len],1,nshift);
else
    shift=randi([options.limit,len-options.limit],1,nshift);
end
aucrand=NaN(size(label,1),nc,nshift);

parfor i = 1:nshift
    % Shift the signal
    shifted = circshift(signal, shift(i), 2);

    % Temporary array to store AUC results for this iteration
    tempAUC = zeros(size(label, 1), nc);

    for b = 1:size(label, 1)
        for c = 1:nc
            % Calculate the AUC
            [~,~,~,auc] = perfcurve(label(b,:), shifted(c,:), 1);
            tempAUC(b, c) = auc;
        end
    end

    % Assign the temporary AUC results to the appropriate slice of aucrand
    aucrand(:,:,i) = tempAUC;
end

function [act,sup,pvalue,nsig,overlap_act,overlap_sup]=rocstats(aucrand,auc,options)
% inputs: aucrand = random shifted roc (beh,cell,shift), auc=original data
% auc (beh,cell), cutoff = cutoff to separate activated or suppressed (0.5 for auc, 0 for mean difference)
arguments
    aucrand
    auc
    options.cutoff = 0.5
end
nshift=size(aucrand,3);
[nb,nc]=size(auc);
pvalue = NaN(nb,nc);
for b=1:nb
    parfor c = 1:nc
        if sum(aucrand(b,c,:)>auc(b,c)) >= nshift/2
            pvalue(b,c)=2*sum(auc(b,c)>=aucrand(b,c,:))./nshift;
        elseif sum(aucrand(b,c,:)>auc(b,c)) < nshift/2
            pvalue(b,c)=2*sum(auc(b,c)<=aucrand(b,c,:))./nshift;
        end
    end
end
pvalue(pvalue==0)=2*0.1/nshift; % if p value = 0 assign the upper bound
nsig(:,1)=sum(pvalue<=0.05&auc>options.cutoff,2);
nsig(:,2)=sum(pvalue<=0.05&auc<options.cutoff,2);% first col activated cell, second col supressed cell  
% number of overlap 
act=(pvalue<=0.05&auc>options.cutoff);
sup=(pvalue<=0.05&auc<options.cutoff);
overlap_act=act*act';
overlap_sup=sup*sup';