% Timing cov calculations to compare with python
function profile_dist_cov(n1, n2, lams, lamz, pu, nreps)

    addpath(genpath('GPMSAmatlab'))

    beta = exp(-0.25 * linspace(0, 1, pu))';
    X1 = rand(n1, pu);
    X2 = rand(n2, pu);

    tic;
    for i = 1:nreps
        genDist(X1, []);
    end
    t = toc;
    fprintf('create square dist x%d %0.5g s\n', nreps, t);

    gd = genDist(X1, []);
    tic;
    for i = 1:nreps
        gCovMat(gd,beta,lamz,lams);
    end
    t = toc;
    fprintf('create square cov x%d %0.5g s\n', nreps, t);

    tic;
    for i = 1:nreps
        genDist2(X1, X2, []);
    end
    t = toc;
    fprintf('create rect dist x%d %0.5g s\n', nreps, t);

    gd = genDist2(X1, X2, []);
    tic;
    for i = 1:nreps
        gCovMat(gd,beta,lamz,lams);
    end
    t = toc;
    fprintf('create rect cov x%d %0.5g s\n', nreps, t);

end
