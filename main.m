clear; 
clc;
close all;
rng(1);   % reproducibility of the FHS simulations
% Requires the Econometrics Toolbox (garch, gjr, estimate)

% output folders (generated files, not tracked by git)
OutDir = 'output';
FigDir = fullfile(OutDir, 'figures');
TabDir = fullfile(OutDir, 'tables');
if ~exist(FigDir, 'dir'), mkdir(FigDir); end
if ~exist(TabDir, 'dir'), mkdir(TabDir); end

% save the console output to a log file
LogFile = fullfile(OutDir, 'log.txt');
if exist(LogFile, 'file'), delete(LogFile); end
diary(LogFile);

%% 1. DATA
T = readtable('portfolio.csv');
date = T{:,1};
ptfPr = T{:,2:end};
stocks = T.Properties.VariableNames(2:end);

% daily simple returns
Returns = tick2ret(ptfPr);
DateReturns = date(2:end);
SampleSize = size(Returns,1);
NStocks = size(Returns,2);

% check for anomalous returns (possible price adjustment errors)
[rr, col_idx] = find(abs(Returns) > 0.30);
for k = 1:numel(rr)
    fprintf('WARNING: %s on %s has a return of %.1f%%\n', stocks{col_idx(k)}, string(DateReturns(rr(k))), 100*Returns(rr(k),col_idx(k)));
end

%% 2. PORTFOLIO
W = [0.056, 0.061, 0.080, 0.041, 0.183, 0.044, 0.052, 0.138, 0.138, 0.152, 0.055];
assert(numel(W) == NStocks, 'The number of weights does not match the number of stocks');
assert(abs(sum(W) - 1) < 1e-9, 'The weights do not sum to 1');
PortfolioReturns = Returns * W(:);

%% 3. PARAMETERS
WS = 250;
pVaR = [0.05 0.01];
H = 1;          % VaR horizon set to 1 day
nSim = 10000;   % number of simulated paths for the FHS
refit = 5;      % GARCH/GJR models are re-estimated every 5 days (for computation time)

% Test window definition (covers the entire interval required for the two regimes)
TestWindowStart = find(DateReturns >= datetime('01-Jan-2013'), 1, "first"); 
TestWindowEnd   = find(DateReturns <= datetime('31-Dec-2021'), 1, "last");
TestWindowStart = max(TestWindowStart, WS+1);
TestWindow      = TestWindowStart:TestWindowEnd;
nT              = numel(TestWindow);
TestDates       = DateReturns(TestWindow);

fprintf('Overall test from %s to %s, %d observations\n\n', string(TestDates(1)), string(TestDates(end)), nT);

%% 5. 1-DAY VaR ESTIMATION
Normal     = zeros(nT,2);   
Hist       = zeros(nT,2);
FHS_P_G    = zeros(nT,2);   
FHS_P_J    = zeros(nT,2);   
ES_P_G     = zeros(nT,2);
ES_P_J     = zeros(nT,2);
PIT        = zeros(nT,4);   % forecast CDF at the realized return: Normal, HS, FHS-GARCH, FHS-GJR

for t = TestWindow
    i = t - TestWindowStart + 1;
    idx = t-WS:t-1;
    X = PortfolioReturns(idx);

    % periodic re-estimation of the parameters (in % for numerical stability)
    if mod(i-1, refit) == 0
        fprintf('Re-estimating GARCH/GJR models: %s (%d/%d)\n', string(DateReturns(t)), i, nT);
        P_port.garch = fitParams(100*X, 'GARCH');
        P_port.gjr   = fitParams(100*X, 'GJR');
    end

    % Parametric (normal, zero mean)
    Sigma = std(X);
    Normal(i,:) = -norminv(pVaR) * Sigma;   

    % Historical simulation
    Hist(i,:) = -quantile(X, pVaR);

    % Portfolio FHS, GARCH(1,1) filter
    [FHS_P_G(i,:), ES_P_G(i,:), RpG] = fhsVaR(P_port.garch, 100*X, H, nSim, pVaR, 1);

    % Portfolio FHS, GJR-GARCH(1,1) filter
    [FHS_P_J(i,:), ES_P_J(i,:), RpJ] = fhsVaR(P_port.gjr, 100*X, H, nSim, pVaR, 1);

    % PIT of the realized return under each 1-day forecast distribution
    r = PortfolioReturns(t);
    PIT(i,:) = [normcdf(r/Sigma), pitECDF(X, r), pitECDF(RpG, r), pitECDF(RpJ, r)];
end

%% 5b. MONTE CARLO STABILITY OF THE FHS-GJR ESTIMATE
fprintf('\nAssessing Monte Carlo stability...\n');
nSimGrid = [1000 5000 10000 20000];
MCStd = zeros(numel(nSimGrid), 2);
for g = 1:numel(nSimGrid)
    n = nSimGrid(g);
    v = zeros(50,2);
    for s = 1:50
        v(s,:) = fhsVaR(P_port.gjr, 100*X, H, n, pVaR, 1);
    end
    MCStd(g,:) = std(v);
    fprintf('nSim=%6d  std VaR95=%.4f  std VaR99=%.4f\n', n, MCStd(g,1), MCStd(g,2));
end
writetable(table(nSimGrid(:), MCStd(:,1), MCStd(:,2), ...
    'VariableNames', {'nSim', 'Std_VaR95', 'Std_VaR99'}), fullfile(TabDir, 'mc_stability.csv'));

%% 6. GLOBAL MATRICES SETUP AND REGIME DEFINITION
ActualReturns = PortfolioReturns(TestWindow);

VaRMatrixFull = [Normal(:,1), Hist(:,1), FHS_P_G(:,1), FHS_P_J(:,1), ...
                 Normal(:,2), Hist(:,2), FHS_P_G(:,2), FHS_P_J(:,2)];

% Logical vectors to isolate the two regimes
IndStable  = (TestDates >= datetime('01-Jan-2014')) & (TestDates <= datetime('31-Dec-2017'));
IndCrisis  =(TestDates >= datetime('01-Jan-2018')) & (TestDates <= datetime('31-Dec-2021'));

%% 7. FORMAL BACKTESTING: STABLE REGIME (2014 - 2017)
fprintf('\n%s\n', repmat('#', 1, 112));
fprintf('>>> START OF ANALYSIS: STABLE REGIME (01/2014 - 12/2017) <<<\n');
fprintf('%s\n', repmat('#', 1, 112));

[BT_Stable, figStable] = runBacktestAndReport(ActualReturns(IndStable), VaRMatrixFull(IndStable, :), ...
    TestDates(IndStable), H, 0.95, 'Stable Regime (2014-2017)');
writetable(BT_Stable, fullfile(TabDir, 'backtest_stable.csv'));
exportgraphics(figStable, fullfile(FigDir, 'backtest_stable.pdf'), 'ContentType', 'vector');
figStable.Visible = 'on';

%% 8. FORMAL BACKTESTING: CRISIS REGIME (2018 - 2021)
fprintf('\n%s\n', repmat('#', 1, 112));
fprintf('>>> START OF ANALYSIS: CRISIS REGIME (01/2018 - 12/2021) <<<\n');
fprintf('%s\n', repmat('#', 1, 112));

[BT_Crisis, figCrisis] = runBacktestAndReport(ActualReturns(IndCrisis), VaRMatrixFull(IndCrisis, :), ...
    TestDates(IndCrisis), H, 0.95, 'Crisis Regime (2018-2021)');
writetable(BT_Crisis, fullfile(TabDir, 'backtest_crisis.csv'));
exportgraphics(figCrisis, fullfile(FigDir, 'backtest_crisis.pdf'), 'ContentType', 'vector');
figCrisis.Visible = 'on';

%% 9. PLOTS OVER THE ENTIRE SAMPLE
Names  = {'Normal','HS','FHS-P GARCH','FHS-P GJR'};
Models = {Normal, Hist, FHS_P_G, FHS_P_J};

% figures are drawn hidden, exported, and only then shown (avoids the
% window being closed or redrawn while the PDF is being written)
fig = figure('Name', '1-day VaR: all models', 'Visible', 'off');
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = gobjects(numel(Models), 1);
for m = 1:numel(Models)
    ax(m) = nexttile(tl);
    plot(ax(m), TestDates, Models{m}(:,1), 'LineWidth', 1); hold(ax(m), 'on');
    plot(ax(m), TestDates, Models{m}(:,2), 'LineWidth', 1);
    hold(ax(m), 'off'); grid(ax(m), 'on');
    title(ax(m), Names{m});
    legend(ax(m), {'VaR 95%', 'VaR 99%'}, 'Location', 'northwest');
end
title(tl, '1-day VaR: 95% vs 99% comparison by model (Full Sample)');
xlabel(tl, 'Date');
ylabel(tl, '1-day VaR');
linkaxes(ax, 'xy');
exportgraphics(fig, fullfile(FigDir, 'var_overview.pdf'), 'ContentType', 'vector');
fig.Visible = 'on';

% file-safe model tags (e.g. 'FHS-P GARCH' -> 'fhs_p_garch')
Tags = lower(regexprep(Names, '[^A-Za-z0-9]+', '_'));

for m = 1:numel(Models)
    figM = figure('Name', Names{m}, 'Visible', 'off');
    axM = axes('Parent', figM);
    plot(axM, TestDates, Models{m}(:,1), 'LineWidth', 1); hold(axM, 'on');
    plot(axM, TestDates, Models{m}(:,2), 'LineWidth', 1);
    hold(axM, 'off'); grid(axM, 'on');
    xlabel(axM, 'Date'); ylabel(axM, '1-day VaR');
    legend(axM, {'VaR 95%', 'VaR 99%'}, 'Location', 'best');
    title(axM, sprintf('%s: 1-day VaR', Names{m}));
    exportgraphics(figM, fullfile(FigDir, sprintf('var_%s.pdf', Tags{m})), 'ContentType', 'vector');
    figM.Visible = 'on';
end

%% 10. CSV EXPORT OF VaR, ES, VIOLATIONS AND PIT (ALL MODELS)
Lev = round(100*(1 - pVaR));   % [95 99]
Series = table(TestDates(:), ActualReturns, 'VariableNames', {'Date', 'Realized_return'});
for m = 1:numel(Models)
    for l = 1:numel(Lev)
        Series.(sprintf('VaR%d_%s', Lev(l), Tags{m})) = Models{m}(:,l);
    end
end
% ES is only available for the FHS models
ESModels = {ES_P_G, ES_P_J};
for m = 1:numel(ESModels)
    for l = 1:numel(Lev)
        Series.(sprintf('ES%d_%s', Lev(l), Tags{m+2})) = ESModels{m}(:,l);
    end
end
% violation indicator: 1 if the realized loss exceeds the VaR
for m = 1:numel(Models)
    for l = 1:numel(Lev)
        Series.(sprintf('Viol%d_%s', Lev(l), Tags{m})) = double(ActualReturns < -Models{m}(:,l));
    end
end
writetable(Series, fullfile(TabDir, 'var_es_series.csv'));

% PIT values (u) and Normal scores (z = norminv(u); +-Inf outside the forecast support)
PITTable = table(TestDates(:), ActualReturns, 'VariableNames', {'Date', 'Realized_return'});
for m = 1:numel(Tags)
    PITTable.(sprintf('u_%s', Tags{m})) = PIT(:,m);
end
for m = 1:numel(Tags)
    PITTable.(sprintf('z_%s', Tags{m})) = norminv(PIT(:,m));
end
writetable(PITTable, fullfile(TabDir, 'pit.csv'));

diary off;
fprintf('Outputs saved in %s\n', fullfile(pwd, OutDir));

%% LOCAL FUNCTIONS (Risk Management & Diagnostics)

function [Results, figReport] = runBacktestAndReport(ReturnsBT, VaRMatrix, DatesBT, H, TestLevel, PeriodName)
    % Naming and Levels for the 4 remaining Models
    VaRNames = {'Normal_95', 'Hist_95', 'FHS_P_G_95', 'FHS_P_J_95', ...
                'Normal_99', 'Hist_99', 'FHS_P_G_99', 'FHS_P_J_99'};
    Levels = [repmat(0.95, 1, 4), repmat(0.99, 1, 4)];

    % 1. varbacktest initialization
    vbt = varbacktest(ReturnsBT, VaRMatrix, ...
        'PortfolioID', PeriodName, ...
        'VaRID', VaRNames, ...
        'VaRLevel', Levels);

    % 2. Running the Statistical Tests
    tPOF = pof(vbt);
    tCCI = cci(vbt);
    tCC  = cc(vbt);

    % Metadata Extraction
    Models_BT = cellstr(string(tPOF.VaRID));
    VaRLevels = tPOF.VaRLevel;
    Obs       = tPOF.Observations;
    Fails     = tPOF.Failures;
    ExpFails  = Obs .* (1 - VaRLevels);
    nModels   = numel(Models_BT);

    if isdatetime(DatesBT)
        dT = DatesBT;
    else
        dT = datetime(DatesBT, 'ConvertFrom', 'datenum');
    end
    PeriodStr = sprintf('%s - %s', char(dT(1), 'dd/MM/yyyy'), char(dT(end), 'dd/MM/yyyy'));

    Tests(1) = struct('Name', 'Test 1 - Kupiec (POF)', 'DF', 1, 'LR', tPOF.LRatioPOF, 'PValue', tPOF.PValuePOF);
    Tests(2) = struct('Name', 'Test 2 - Christoffersen (CCI)', 'DF', 1, 'LR', tCCI.LRatioCCI, 'PValue', tCCI.PValueCCI);
    Tests(3) = struct('Name', 'Test 3 - Conditional Coverage (CC)', 'DF', 2, 'LR', tCC.LRatioCC, 'PValue', tCC.PValueCC);

    for k = 1:numel(Tests)
        Tests(k).Crit   = chi2inv(TestLevel, Tests(k).DF);
        Tests(k).Reject = Tests(k).LR > Tests(k).Crit;
    end

    % 3. Command Window Report
    lineW = 112;
    fprintf('\n%s\n', repmat('=', 1, lineW));
    fprintf('  VaR BACKTESTING REPORT  |  Portfolio: %s  |  %s  |  Test level: %.0f%%\n', ...
        PeriodName, PeriodStr, TestLevel*100);
    fprintf('%s\n', repmat('=', 1, lineW));
    for k = 1:numel(Tests)
        T = Tests(k);
        fprintf('\n  %s\n', upper(T.Name));
        fprintf('  LR ~ chi2(%d), critical value = %.4f\n', T.DF, T.Crit);
        fprintf('  %s\n', repmat('-', 1, lineW-2));
        fprintf('  %-13s %7s %8s %10s %10s %12s %13s %9s   %-14s\n', ...
            'Model', 'VaR', 'Obs.', 'Exp.viol.', 'Violat.', 'LR stat.', 'Crit. value', 'p-value', 'Outcome');
        fprintf('  %s\n', repmat('-', 1, lineW-2));
        for r = 1:nModels
            if T.Reject(r)
                outcomeStr = 'REJECT H0';
            else
                outcomeStr = 'DO NOT REJECT';
            end
            fprintf('  %-13s %6.0f%% %8d %10.1f %10d %12.4f %13.4f %9.4f   %-14s\n', ...
                Models_BT{r}, VaRLevels(r)*100, Obs(r), ExpFails(r), Fails(r), ...
                T.LR(r), T.Crit, T.PValue(r), outcomeStr);
        end
    end
    fprintf('\n  Rule: H0 is rejected if LR stat. > critical value (p-value < %.2f).\n', 1-TestLevel);
    fprintf('%s\n\n', repmat('=', 1, lineW));

    % Results table (for CSV export)
    Results = table(Models_BT, VaRLevels, Obs, ExpFails, Fails, ...
        Tests(1).LR, Tests(1).PValue, Tests(1).Reject, ...
        Tests(2).LR, Tests(2).PValue, Tests(2).Reject, ...
        Tests(3).LR, Tests(3).PValue, Tests(3).Reject, ...
        'VariableNames', {'Model', 'VaRLevel', 'Observations', 'ExpectedFailures', 'Failures', ...
        'LR_POF', 'PValue_POF', 'Reject_POF', 'LR_CCI', 'PValue_CCI', 'Reject_CCI', ...
        'LR_CC', 'PValue_CC', 'Reject_CC'});

    % 4. Graphical Matrix Report
    critPOF = chi2inv(TestLevel, 1);
    critCCI = chi2inv(TestLevel, 1);
    critCC  = chi2inv(TestLevel, 2);

    colW = [0.14, 0.08, 0.12, 0.13, 0.09, 0.13, 0.09, 0.13, 0.09];
    colW = colW / sum(colW);
    colX = [0 cumsum(colW(1:end-1))];
    colC = colX + colW/2;

    hHeaderGroup = 1.3; hHeaderCols = 1.1; hRow = 0.95; hTitle = 2.2; hFoot = 1.4;
    totH = hTitle + hHeaderGroup + hHeaderCols + nModels*hRow + hFoot;

    figReport = figure('Name', sprintf('VaR Diagnostic Summary - %s', PeriodName), 'NumberTitle', 'off', ...
        'Color', 'w', 'Position', [100 50 1150 550], 'Visible', 'off');  % Height optimized for 8 models
    ax_rep = axes('Parent', figReport, 'Position', [0.03 0.04 0.94 0.92], ...
        'XLim', [0 1], 'YLim', [0 totH], 'YDir', 'reverse', 'Visible', 'off');
    hold(ax_rep, 'on');

    cDark = [0.12 0.14 0.18]; cGrey = [0.45 0.48 0.52]; cLine = [0.82 0.85 0.88];
    cHeader = [0.93 0.95 0.97]; rowAlt = [0.975 0.980 0.985];

    text(ax_rep, 0, 0.8, sprintf('VaR Diagnostic & Backtesting Validation Matrix - %s', PeriodName), ...
        'FontSize', 15, 'FontWeight', 'bold', 'Color', cDark, 'Interpreter', 'none');
    text(ax_rep, 0, 1.6, sprintf('Period: %s (N = %d) | H = %d (Daily) | Test Confidence Level: %.0f%%', ...
        PeriodStr, Obs(1), H, TestLevel*100), 'FontSize', 9, 'Color', cGrey, 'Interpreter', 'none');

    y = hTitle;
    rectangle(ax_rep, 'Position', [0 y colX(4) hHeaderGroup], 'FaceColor', cHeader, 'EdgeColor', 'none');
    rectangle(ax_rep, 'Position', [colX(4) y colW(4)+colW(5) hHeaderGroup], 'FaceColor', [0.88 0.92 0.96], 'EdgeColor', 'none');
    rectangle(ax_rep, 'Position', [colX(6) y colW(6)+colW(7) hHeaderGroup], 'FaceColor', [0.93 0.90 0.95], 'EdgeColor', 'none');
    rectangle(ax_rep, 'Position', [colX(8) y colW(8)+colW(9) hHeaderGroup], 'FaceColor', [0.89 0.94 0.92], 'EdgeColor', 'none');

    text(ax_rep, colX(4)/2, y + hHeaderGroup/2, 'BASIC DATA', 'FontWeight', 'bold', 'FontSize', 8.5, 'Color', cDark, 'HorizontalAlignment', 'center');
    text(ax_rep, colX(4) + (colW(4)+colW(5))/2, y + hHeaderGroup/2, sprintf('KUPIEC (POF) [LR > %.2f]', critPOF), 'FontWeight', 'bold', 'FontSize', 8.5, 'Color', [0.12 0.28 0.44], 'HorizontalAlignment', 'center');
    text(ax_rep, colX(6) + (colW(6)+colW(7))/2, y + hHeaderGroup/2, sprintf('CHRISTOFFERSEN (CCI) [LR > %.2f]', critCCI), 'FontWeight', 'bold', 'FontSize', 8.5, 'Color', [0.30 0.20 0.40], 'HorizontalAlignment', 'center');
    text(ax_rep, colX(8) + (colW(8)+colW(9))/2, y + hHeaderGroup/2, sprintf('COND. COVERAGE (CC) [LR > %.2f]', critCC), 'FontWeight', 'bold', 'FontSize', 8.5, 'Color', [0.12 0.35 0.28], 'HorizontalAlignment', 'center');

    y = y + hHeaderGroup;
    colLabels = {'Model', 'VaR', 'Fails / Exp', 'LR (p-val)', 'Outcome', 'LR (p-val)', 'Outcome', 'LR (p-val)', 'Outcome'};
    for c = 1:numel(colLabels)
        text(ax_rep, colC(c), y + hHeaderCols/2, colLabels{c}, 'FontSize', 8.5, 'FontWeight', 'bold', 'Color', cDark, 'HorizontalAlignment', 'center', 'Interpreter', 'none');
    end
    y = y + hHeaderCols;
    plot(ax_rep, [0 1], [y y], 'Color', cDark, 'LineWidth', 1.2);

    for r = 1:nModels
        yc = y + hRow/2;
        if mod(r, 2) == 0, rectangle(ax_rep, 'Position', [0 y 1 hRow], 'FaceColor', rowAlt, 'EdgeColor', 'none'); end
        if r > 1 && VaRLevels(r) ~= VaRLevels(r-1), plot(ax_rep, [0 1], [y y], 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5); end
        
        text(ax_rep, colC(1), yc, Models_BT{r}, 'FontWeight', 'bold', 'FontSize', 9, 'Color', cDark, 'HorizontalAlignment', 'center', 'Interpreter', 'none');
        text(ax_rep, colC(2), yc, sprintf('%.0f%%', VaRLevels(r)*100), 'FontSize', 9, 'Color', cDark, 'HorizontalAlignment', 'center');
        text(ax_rep, colC(3), yc, sprintf('%d / %.1f', Fails(r), ExpFails(r)), 'FontSize', 9, 'Color', cDark, 'HorizontalAlignment', 'center');
        
        % Test POF
        rejPOF = tPOF.LRatioPOF(r) > critPOF;
        if isnan(tPOF.LRatioPOF(r)), pStr = 'NaN'; elseif tPOF.PValuePOF(r) < 0.001, pStr = '<0.001'; else, pStr = sprintf('%.3f', tPOF.PValuePOF(r)); end
        text(ax_rep, colC(4), yc, sprintf('%.2f (%s)', tPOF.LRatioPOF(r), pStr), 'FontSize', 8.5, 'Color', cDark, 'HorizontalAlignment', 'center');
        drawStatusPill(ax_rep, colX(5), colW(5), y, hRow, rejPOF);
        
        % Test CCI
        rejCCI = tCCI.LRatioCCI(r) > critCCI;
        if isnan(tCCI.LRatioCCI(r)), pStr = 'NaN'; elseif tCCI.PValueCCI(r) < 0.001, pStr = '<0.001'; else, pStr = sprintf('%.3f', tCCI.PValueCCI(r)); end
        text(ax_rep, colC(6), yc, sprintf('%.2f (%s)', tCCI.LRatioCCI(r), pStr), 'FontSize', 8.5, 'Color', cDark, 'HorizontalAlignment', 'center');
        drawStatusPill(ax_rep, colX(7), colW(7), y, hRow, rejCCI);
        
        % Test CC
        rejCC = tCC.LRatioCC(r) > critCC;
        if isnan(tCC.LRatioCC(r)), pStr = 'NaN'; elseif tCC.PValueCC(r) < 0.001, pStr = '<0.001'; else, pStr = sprintf('%.3f', tCC.PValueCC(r)); end
        text(ax_rep, colC(8), yc, sprintf('%.2f (%s)', tCC.LRatioCC(r), pStr), 'FontSize', 8.5, 'Color', cDark, 'HorizontalAlignment', 'center');
        drawStatusPill(ax_rep, colX(9), colW(9), y, hRow, rejCC);
        
        y = y + hRow;
    end
    plot(ax_rep, [0 1], [y y], 'Color', cLine, 'LineWidth', 1);
end

%% LOCAL FUNCTIONS (GARCH Estimation and FHS)

function P = fitParams(Y, type)
n = size(Y,2);
P.K = zeros(1,n); P.G = zeros(1,n); P.A = zeros(1,n); P.L = zeros(1,n);
for j = 1:n
    if strcmp(type,'GARCH')
        Mdl = garch(1,1);
    else
        Mdl = gjr(1,1);
    end
    Est = estimate(Mdl, Y(:,j), 'Display', 'off');
    P.K(j) = Est.Constant;
    P.G(j) = Est.GARCH{1};
    P.A(j) = Est.ARCH{1};
    if strcmp(type,'GJR')
        P.L(j) = Est.Leverage{1};   
    end
end
end

function V = filterVar(P, Y)
[T,n] = size(Y);
V = zeros(T,n);
V(1,:) = var(Y);
for k = 2:T
    e = Y(k-1,:);
    V(k,:) = P.K + P.G.*V(k-1,:) + (P.A + P.L.*(e<0)).*e.^2;
end
end

function [VaR, ES, Rp] = fhsVaR(P, Y, H, nSim, pVaR, w)
V = filterVar(P, Y);
Z = Y ./ sqrt(V);                                   
eLast = Y(end,:);
s2 = P.K + P.G.*V(end,:) + (P.A + P.L.*(eLast<0)).*eLast.^2;   
s2 = repmat(s2, nSim, 1);
cum = ones(nSim, size(Y,2));
for h = 1:H
    j = randi(size(Z,1), nSim, 1);                  
    e = sqrt(s2) .* Z(j,:);                         
    cum = cum .* (1 + e/100);                       
    s2 = P.K + P.G.*s2 + (P.A + P.L.*(e<0)).*e.^2;  
end
Rp = (cum - 1) * w(:);                              
q = quantile(Rp, pVaR);
VaR = -q;
ES = zeros(1, numel(pVaR));
for k = 1:numel(pVaR)
    ES(k) = -mean(Rp(Rp <= q(k)));
end
end

function u = pitECDF(S, r)
% mid-point empirical CDF of the sample S evaluated at r (ties count one half)
u = (sum(S < r) + 0.5*sum(S == r)) / numel(S);
end

function drawStatusPill(ax, x, w, y, h, isReject)
    if isReject
        bg = [0.98 0.88 0.88]; fg = [0.70 0.15 0.15]; label = 'REJ';
    else
        bg = [0.88 0.95 0.90]; fg = [0.10 0.48 0.22]; label = 'OK';
    end
    pw = w * 0.72; ph = h * 0.65;
    px = x + (w - pw)/2; py = y + (h - ph)/2;
    rectangle(ax, 'Position', [px py pw ph], 'FaceColor', bg, 'EdgeColor', fg, 'LineWidth', 0.8, 'Curvature', [0.4 0.6]);
    text(ax, px + pw/2, py + ph/2, label, 'Color', fg, 'FontSize', 8, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end