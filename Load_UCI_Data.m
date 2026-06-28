function [feat, label] = Load_UCI_Data(DataSet_Index)

if nargin == 0
    clc
    close all
    clear all
    addpath('Dataset\')
    DataSet_Index = 1;
end

% This study uses 20 datasets from the UCI Machine Learning Repository
% (https://archive.ics.uci.edu/ml/datasets.php), indexed below by their
% original database index. Only the cases used in this study are retained.
switch DataSet_Index

    case 1
        % Breast Cancer Wisconsin (Original) Data Set
        % https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
        disp('Loading DS1 - Breast Cancer Wisconsin (Original) Data Set')
        fileID = fopen('breast-cancer-wisconsin.data');
        format = repmat('%f', [1, 11]);
        C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
        fclose(fileID);
        feat = C{1,1}(1:end,2:end-1); %row, column
        label1 = C{1,1}(:,end);
        index_2 = find(label1 == 2);
        index_4 = find(label1 == 4);
        label(index_2,1) = 0;
        label(index_4,1) = 1;

    case 2
        % Dermatology Data Set
        % https://archive.ics.uci.edu/ml/datasets/Dermatology
        disp('Loading DS 2 - Dermatology Data Set')
        fileID = fopen('dermatology.data');
        format = repmat('%f', [1, 35]);
        C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
        fclose(fileID);
        feat = C{1,1}(:, 1:end-1);
        label = C{1,1}(:, end);

    case 3
        % Glass Identification Data Set
        % https://archive.ics.uci.edu/ml/datasets/Glass+Identification
        disp('Loading DS 3 - Glass Identification Data Set')
        fileID = fopen('glass.data');
        format = repmat('%f', [1, 11]);
        C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
        fclose(fileID);
        feat = C{1,1}(:, 2:end-1); %exclude ID at column 1
        label = C{1,1}(:, end);

    case 4
        % Lung Cancer Data Set
        % https://archive.ics.uci.edu/ml/datasets/Lung+Cancer
        disp('Loading DS 4 - Lung Cancer Data Set')
        fileID = fopen('lung-cancer.data');
        format = repmat('%f', [1, 57]);
        C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
        fclose(fileID);
        feat = C{1,1}(:, 2:end);
        label = C{1,1}(:, 1); % Attribute 1 is class label

    case 5
        % Statlog (Heart) Data Set
        % https://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29
        disp('Loading DS 5 - Statlog (Heart) Data Set')
        fileID = fopen('heart.dat');
        format = repmat('%f', [1, 14]);
        C = textscan(fileID, format, 'Delimiter',' ', 'CollectOutput', 1);
        fclose(fileID);
        feat = C{1,1}(:, 1:end-1);
        label = C{1,1}(:, end);

    case 6
        % Iris Data Set
        % https://archive.ics.uci.edu/ml/datasets/iris
        % Built in dataset in Matlab
        disp('Loading DS 6 - Iris Data Set')
        load fisheriris.mat
        feat = meas;
        [Total_Data, ~] = size(feat);
        label = zeros(Total_Data,1);
        for i = 1:Total_Data
            if isequal(species{i},'setosa')
                label(i)= 1;
            elseif isequal(species{i},'versicolor')
                label(i)= 2;
            elseif isequal(species{i},'virginica')
                label(i)= 3;
            end
        end

    case 7
        % Arrhythmia Data Set
        % https://archive.ics.uci.edu/ml/datasets/arrhythmia
        % Built in dataset in Matlab
        disp('Loading DS 7- Arrhythmia Data Set')
        load arrhythmia.mat
        feat = X;
        label = Y;

    case 8
        % Ovarian Data Set
        % Built in dataset in Matlab
        disp('Loading DS 8 - Ovarian Data Set')
        load ovariancancer.mat
        feat = obs;
        [Total_Data, ~] = size(feat);
        label = zeros(Total_Data,1);
        for i = 1:Total_Data
            if isequal(grp{i},'Cancer')
                label(i)= 1;
            elseif isequal(grp{i},'Normal')
                label(i)= 0;
            end
        end

    case 9
        % Echocardiogram Data Set
        % https://archive.ics.uci.edu/ml/datasets/Echocardiogram
        disp('Loading DS 9 - Echocardiogram Data Set')
        fileID = fopen('echocardiogram.data');
        format = [repmat('%s', [1, 13])]; % 13 Row in the table
        C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
        fclose(fileID);
        c1 = C{1,1}(1:end,[1:7,9,13]); % Remove meaningless attribute
        c1(51,:) = [];
        j=1;
        for i = 1:132 % Remove unknowmn value
            test = find(strcmp(c1(i,:), '?'));
            if numel(test) == 0
                c2(j,:)=c1(i,:);
                j=j+1;
            else
            end
        end
        for i = 1:61 % Convert character array to numeric array
            for j = 1:8
                feat(i,j) = str2num(c2{i,j});
            end
            label(i,:) = str2num(c2{i,end});
        end

    case 10
        % Liver Disorders Data Set
        % https://archive.ics.uci.edu/ml/datasets/Liver+Disorders
        disp('Loading DS 10 - Liver Disorders Data Set')
        fileID = fopen('bupa.data');
        format = [repmat('%f', [1, 7])];
        C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
        fclose(fileID);
        feat = C{1,1}(1:end, 1:end-1);
        label = C{1,1}(1:end, end);

    case 11
        % Connectionist Bench (Sonar, Mines vs. Rocks) Data Set
        % https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar%2C+Mines+vs.+Rocks%29
        disp('Loading DS 11 - Connectionist Bench (Sonar, Mines vs. Rocks) Data Set')
        fileID = fopen('sonar.all-data');
        format = [repmat('%f', [1, 60]) '%s'];
        C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
        fclose(fileID);
        feat = C{1,1};
        label1 = C{1,2};
        [Total_Data, ~] = size(feat);
        label = zeros(Total_Data,1);
        for i=1:Total_Data
            if label1{i} == 'M'        % Mines
                label(i,:) = 1;
            else % if label1{i} == 'R' % Rock
                label(i,:) = 0;
            end
        end

    case 12
        % Parkinsons Data Set
        % https://archive.ics.uci.edu/ml/datasets/Parkinsons
        disp('Loading DS 12 - Parkinsons Data Set')
        fileID = fopen('parkinsons.data');
        format = ['%s' repmat('%f', [1, 23])];
        C = textscan(fileID, format, 'Delimiter',',', 'headerlines', 1, 'CollectOutput', 1);
        fclose(fileID);
        feat = C{1,2}(1:end, 1:end ~= 17);
        label = C{1,2}(1:end, 17);

    case 13
        % Waveform Database Generator (Version 1) Data Set
        % https://archive.ics.uci.edu/ml/datasets/Waveform+Database+Generator+%28Version+1%29
        disp('Loading DS 13 - Waveform Database Generator (Version 1) Data Set')
        fileID = fopen('waveform.data');
        format = [repmat('%f', [1, 22])];
        C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
        fclose(fileID);
        feat = C{1,1}(1:end, 1:end-1);
        label = C{1,1}(1:end, end);

    case 14
        % Yeast Data Set
        % https://archive.ics.uci.edu/ml/datasets/Yeast
        disp('Loading DS 14 - Yeast Data Set')
        fileID = fopen('yeast.data');
        format = [ '%s' repmat('%f', [1, 8]) '%s'];
        C = textscan(fileID, format, 'EmptyValue', -Inf, 'CollectOutput', 1);
        fclose(fileID);
        feat = C{1,2};
        [Total_Data, ~] = size(feat);
        label = zeros(Total_Data,1);
        for i = 1:Total_Data
            if isequal(C{1,3}{i},'CYT')
                label(i)= 1;
            elseif isequal(C{1,3}{i},'ERL')
                label(i)= 2;
            elseif isequal(C{1,3}{i},'EXC')
                label(i)= 3;
            elseif isequal(C{1,3}{i},'ME1')
                label(i)= 4;
            elseif isequal(C{1,3}{i},'ME2')
                label(i)= 5;
            elseif isequal(C{1,3}{i},'ME3')
                label(i)= 6;
            elseif isequal(C{1,3}{i},'MIT')
                label(i)= 7;
            elseif isequal(C{1,3}{i},'NUC')
                label(i)= 8;
            elseif isequal(C{1,3}{i},'POX')
                label(i)= 9;
            elseif isequal(C{1,3}{i},'VAC')
                label(i)= 10;
            end
        end

    case 15
        % Blood Transfusion Service Center Data Set
        % https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
        disp('Loading DS 15 - Blood Transfusion Service Center Data Set')
        fileID = fopen('transfusion.data');
        format = [repmat('%f', [1, 5])];
        C = textscan(fileID, format, 'Delimiter',',', 'headerlines', 1, 'CollectOutput', 1);
        fclose(fileID);
        feat = C{1,1}(1:end, 1:end-1);
        label = C{1,1}(1:end, 5);

    case 16
        % Wine Data Set
        % https://archive.ics.uci.edu/ml/datasets/Wine
        disp('Loading DS 16 - Wine Data Set')
        fileID = fopen('wine.data');
        format = [repmat('%f', [1, 14])];
        C = textscan(fileID, format, 'Delimiter', ',' , 'CollectOutput', 1);
        fclose(fileID);
        feat = C{1,1}(1:end, 2:end);
        label = C{1,1}(1:end, 1);

    case 17
        % Semeion Handwritten Digit Data Set
        % https://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit
        disp('Loading DS 17 - Semeion Handwritten Digit Data Set')
        fileID = fopen('semeion.data');
        format = [repmat('%f', [1, 266])];
        C = textscan(fileID, format, 'EmptyValue', -Inf, 'CollectOutput', 1);
        fclose(fileID);
        feat = C{1,1}(1:end, 1:end-10);
        [~,label]=max(C{1,1}(1:end, 257:end),[],2); % Find '1' column value in every row

    case 18
        % DARWIN Data Set
        % https://archive.ics.uci.edu/dataset/732/darwin
        disp('Loading DS 18 - Darwin Data Set')
        C = readtable('DARWIN.csv'); % Stores data in a table
        feat = table2array(C(1:end, 2:end-1));
        [Total_Data, ~] = size(feat);
        label_Temp = C(1:end, end);
        label = zeros(Total_Data,1);
        for i = 1:Total_Data
            if isequal(label_Temp{i, 1}, {'P'})
                label(i) = 1;
            elseif isequal(label_Temp{i, 1}, {'H'})
                label(i) = 2;
            end
        end

    case 19
        % Person Classification Gait Data Set
        % https://archive.ics.uci.edu/dataset/561/person+classification+gait+data
        disp('Loading DS 19 - Person Classification Gait Data Set')
        filename = "PersonGaitDataSet.mat";
        myVars = {"X","Y"};
        S = load(filename,myVars{:});
        feat = S.X;
        label = S.Y;

    case 20
        % Gastrointestinal Lesions in Regular Colonoscopy Data Set
        % https://archive.ics.uci.edu/dataset/408/gastrointestinal+lesions+in+regular+colonoscopy
        disp('Loading DS 20 - Gastrointestinal Lesions in Regular Colonoscopy Data Set')
        fileID = fopen('gastrointestinal+lesions+in+regular+colonoscopy.txt');
        format = [repmat('%f', [1, 152])];
        C = textscan(fileID, format, 'Delimiter', ',' , 'headerlines', 1, 'CollectOutput', 1);
        fclose(fileID);
        feat = (C{1,1}(3:end, 1:end)).';
        label = (C{1,1}(1, 1:end)).';

end
end