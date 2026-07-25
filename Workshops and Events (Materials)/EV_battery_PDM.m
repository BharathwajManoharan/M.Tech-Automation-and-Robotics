%% EV Battery Predictive Maintenance Pipeline (Multi-Model Comparison)
% Compares Random Forest, SVM, Neural Net, and LSTM models
% on simulated EV battery data from Simulink.

clc; clear; close all;

%% 1. Run the Simulink Model
disp('Running Simulink model...');
simOut = sim('EV_Battery_Model','ReturnWorkspaceOutputs','on');
simData = simOut.simData;

%% 2. Extract Data from simData (Bus signals)
time = simData.time;
busVals = simData.signals.values;   % [N x 4]

Voltage = busVals(:,1);
Current = busVals(:,2);
SoC     = busVals(:,3);
Temp    = busVals(:,4);

%% 3. Preprocess Features
HealthIndex = normalize(Voltage - 0.002*abs(Current));
RUL = max(time) - time;   

% Feature table (for classical ML models)
T = table(Voltage, Current, SoC, Temp, HealthIndex, RUL);

%% 4. Define models to compare
modelList = {'RF','SVM','NN','LSTM'};
results = [];

%% 5. Loop through models
for i = 1:numel(modelList)
    modelChoice = modelList{i};
    fprintf('\n=== Training %s ===\n', modelChoice);

    switch modelChoice
        case 'RF'
            cv = cvpartition(height(T),'HoldOut',0.3);
            trainTbl = T(training(cv),:);
            testTbl  = T(test(cv),:);

            Mdl = fitrensemble(trainTbl(:,1:4), trainTbl.RUL, ...
                'Method','Bag','NumLearningCycles',50);

            yPred = predict(Mdl, testTbl(:,1:4));
            rmse = sqrt(mean((yPred - testTbl.RUL).^2));

            yAllPred = predict(Mdl, T(:,1:4));

        case 'SVM'
            cv = cvpartition(height(T),'HoldOut',0.3);
            trainTbl = T(training(cv),:);
            testTbl  = T(test(cv),:);

            Mdl = fitrsvm(trainTbl(:,1:4), trainTbl.RUL, ...
                'KernelFunction','gaussian');

            yPred = predict(Mdl, testTbl(:,1:4));
            rmse = sqrt(mean((yPred - testTbl.RUL).^2));

            yAllPred = predict(Mdl, T(:,1:4));

        case 'NN'
            X = table2array(T(:,1:4))';
            Y = RUL';

            net = feedforwardnet(10);  % 10 hidden neurons
            net = train(net, X, Y);

            yAllPred = net(X)';
            rmse = sqrt(mean((yAllPred - RUL).^2));

        case 'LSTM'
            X = [Voltage Current SoC Temp]';
            Y = RUL';

            XTrain = {X};
            YTrain = {Y};

            inputSize = 4;
            numHiddenUnits = 50;
            numResponses = 1;

            layers = [ ...
                sequenceInputLayer(inputSize)
                lstmLayer(numHiddenUnits,'OutputMode','sequence')
                fullyConnectedLayer(numResponses)
                regressionLayer];

            options = trainingOptions('adam', ...
                'MaxEpochs',20, ...
                'GradientThreshold',1, ...
                'Verbose',false, ...
                'Plots','none');

            net = trainNetwork(XTrain,YTrain,layers,options);

            % ✅ FIX: unwrap cell prediction
            YPredCell = predict(net, XTrain);
            yAllPred = YPredCell{1}';
            rmse = sqrt(mean((yAllPred - RUL).^2));
    end

    % Store results
    results = [results; {modelChoice, rmse, yAllPred}];

    % Individual plots
    figure;
    plot(time, RUL,'-b','LineWidth',1.5); hold on;
    plot(time, yAllPred,'--r','LineWidth',1.5);
    xlabel('Time (s)'); ylabel('Remaining Useful Life');
    legend('True RUL','Predicted RUL');
    title(['RUL Prediction using ' modelChoice]);
    grid on;
end

%% 6. Summary Table
resultsTable = cell2table(results, ...
    'VariableNames',{'Model','RMSE','PredictedRUL'});
disp('=== Model Comparison Summary ===');
disp(resultsTable);

%% 7. Overlay Plot for Comparison
figure;
plot(time, RUL,'-k','LineWidth',2); hold on;

colors = lines(numel(modelList));
for i = 1:height(resultsTable)
    plot(time, resultsTable.PredictedRUL{i}, ...
        '--','Color',colors(i,:),'LineWidth',1.5);
end

xlabel('Time (s)'); ylabel('Remaining Useful Life');
legend(['True RUL', resultsTable.Model']);
title('Overlay: RUL Predictions by All Models');
grid on;
