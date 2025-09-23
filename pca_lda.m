%% 
function [pdist,score,W] = pca_lda(M,L,n_trial,options)
% Run pca - lda to determine the top linear dimensions that maximally separates
% the classes
% Inputs:
%   M - neural activity (cell x bout), 
% ! NOTE: this function assumes that bout has been sorted such that, the first n bout is the first class, etc. 
%   L - class label (1 x bout)
%   n_trial - number of trials in each class (1 x class)

% Optional inputs:   
arguments
    M
    L
    n_trial
    options.titles = {'huddle', 'rest', 'sniff'}
    options.nPC = 10 % the top n PCs to take for lda 
end
% outputs:
%   pdist: distance matrix within/between classes ( class x class) 
%   score: lda score
%   W: lda weights 

    assert(length(options.titles)==length(unique(L)),'number of class does not match labels')
    n_class = length(options.titles);
    %% perform LDA
    [coeff,score,~,~,~,mu] = pca(M');
    [score,W] = lda(score(:,1:options.nPC),L);
    %% calculate pairwise dist 
    cumn = cumsum(n_trial);
    cumn = [0,cumn];
    pdist = [];
    for i=1:n_class
        for j = 1:n_class
            m = pdist2(score(cumn(i)+1:cumn(i+1),1:3),score(cumn(j)+1:cumn(j+1),1:3));
            if i == j
                mask = triu(true(size(m)),1); % for within class distance, exclude diagonal (same bout to same bout)
            else
                mask = true(size(m));
            end
            pdist(i,j) = mean(m(mask),'all');
        end
    end
    %% make plot 
    figure
    p = [];
    for i=1:n_class
        p(i) = scatter(score(cumn(i)+1:cumn(i+1),1),score(cumn(i)+1:cumn(i+1),2),20,cols(i,:),'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5);
        hold on
        x = mean(score(cumn(i)+1:cumn(i+1),1));
        y = mean(score(cumn(i)+1:cumn(i+1),2));
        stx = std(score(cumn(i)+1:cumn(i+1),1)); sty = std(score(cumn(i)+1:cumn(i+1),2));
        errorbar(x, y, sty,sty,stx, stx,  '-o', 'Color', cols(i,:), 'CapSize', 0);
        %meanbout(a,i,1:2) = mean(score(cumn(i)+1:cumn(i+1),1:2));
    end
    xlabel('LD1');ylabel('LD2');set(gca,'linewidth',1)
    legend(p,titles,'box','off','Interpreter','none','location','bestoutside'); box off
end
%%
function [Y,W] = lda(X,L)
Mdl = fitcdiscr(X,L);
[W, LAMBDA] = eig(Mdl.BetweenSigma, Mdl.Sigma);
lambda = diag(LAMBDA);
[lambda, SortOrder] = sort(lambda, 'descend');
W = W(:, SortOrder);
Y = X*W;
end