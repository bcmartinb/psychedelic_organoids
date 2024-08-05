% Define the directory to save HDF5 files
output_directory = "C:\Users\david\Documents\Voytek Research\LFP_psych_proj\broadband_squared_dataset";
raw_data_dir = "D:\2024-08-01 broadband test for blanca\broadband processor\118-5923\BIPOLES_GR_20240801_alternate orange and blue(000)_BroadbandProcessor.raw";
% Create the directory if it doesn't exist
if ~isfolder(output_directory)
    mkdir(output_directory);
end
all_data = AxisFile(raw_data_dir).RawVoltageData;
broadband = all_data(1,2);
disp(broadband)
lfp_data = broadband.LoadData;
sizes = size(lfp_data);
lfp_rows = sizes(1);
lfp_cols = sizes(2);

% Initialize a cell array to store data for each well
all_wells_data = cell(lfp_rows, lfp_cols);

% Loop through each well
for i = 1:lfp_rows
    for j = 1:lfp_cols
        % Extract the data for the current well
        current_well_data = lfp_data(i, j);
        current_well_data = current_well_data{1,1}.Data;
        
        % Check if the current_well_data is empty
        if isempty(current_well_data)
            disp(['No data for well ' char('A' + i - 1) num2str(j)]);
            continue;
        end
        
        % Ensure current_well_data is numeric
        numeric_data = double(current_well_data);
        
        % Store the numeric_data in the cell array
        all_wells_data{i, j} = numeric_data;
        
        % Plot the electrode data
        figure;
        plot(numeric_data);
        title(sprintf('Well %s%d Electrode Data', char('A' + i - 1), j));
        xlabel('Time');
        ylabel('Voltage');
    end
end

% Convert the cell array to a numeric array if possible, otherwise save as a cell
try
    all_wells_data_numeric = cell2mat(all_wells_data);
    save_data = all_wells_data_numeric;
    datatype = 'double';
catch
    save_data = all_wells_data;
    datatype = 'cell';
end

% Save the combined data to a single HDF5 file
filename = fullfile(output_directory, 'combined_lfp_data.h5');
if exist(filename, 'file')
    delete(filename); % Delete the existing file to avoid conflicts
end

if strcmp(datatype, 'double')
    h5create(filename, '/all_wells_data', size(save_data), 'Datatype', 'double');
else
    h5create(filename, '/all_wells_data', size(save_data), 'Datatype', 'cell');
end
h5write(filename, '/all_wells_data', save_data);
