

#PART ONE: Data Ingestion and Manipulation

#Importing Necessary Libraries
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

def process_worldbank_data(filename):
    # Read the data into a pandas dataframe
    program_dataframe = pd.read_csv(filename)

    # Transpose the dataframe and clean it
    transposed_program_dataframe = program_dataframe.T
    transposed_program_dataframe.columns = transposed_program_dataframe.iloc[0]
    transposed_program_dataframe = transposed_program_dataframe[1:]

    return program_dataframe, transposed_program_dataframe

def explore_most_appearing_indicators(dataframe, countries, top_n=5):
    most_appearing_indicators = {}

    for country in countries:
        # Subset the dataframe for the current country
        country_program_dataframe = dataframe[dataframe['Country Code'] == country]

        # Count the occurrences of each indicator
        indicator_counts = country_program_dataframe['Indicator Code'].value_counts()

        # Select the top N indicators
        top_indicators = indicator_counts.head(top_n).index.tolist()

        # Store the top indicators for the current country
        most_appearing_indicators[country] = top_indicators

    return most_appearing_indicators

#PART TWO: Exploring Statistical Properties

def explore_statistics(dataframe, indicators, countries):
    # Subset the dataframe with selected indicators and countries
    subset_program_dataframe = dataframe[dataframe['Country Code'].isin(countries) & dataframe['Indicator Code'].isin(indicators)]

    # Calculate summary statistics using .describe() and other statistical methods
    summary_stats = subset_program_dataframe.describe()

        # Print summary statistics
    print("\nSummary Statistics:")
    print(summary_stats)

    return summary_stats

#PART THREE: Exploring Correlations

def explore_correlations(dataframe, indicators, countries):
    # Subset the project dataframe with selected indicators and countries
    subset_program_dataframe = dataframe[dataframe['Country Code'].isin(countries) & dataframe['Indicator Code'].isin(indicators)]

    # Extract columns containing indicator values
    value_columns = subset_program_dataframe.columns[5:]

    # Pivot the dataframe to have indicators as rows
    pivot_program_dataframe = subset_program_dataframe.melt(id_vars=['Country Code', 'Country Name', 'Indicator Code'], value_vars=value_columns)

    # Create a pivot table with indicators as columns and fill NaN values with 0
    pivot_table = pivot_program_dataframe.pivot_table(index=['Country Code', 'Country Name'], columns='Indicator Code', values='value', fill_value=0)

    # Calculate correlations
    correlation_matrix = pivot_table.corr()

    # Set the style for better visualization
    sns.set(style="whitegrid")

    # Create a heatmap with a different colormap (viridis)
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f')  # Change 'coolwarm' to 'viridis'

    # A cluster of warm-colored cells suggests a positive correlation between those indicators.
    # Conversely, cool-colored cells indicate a negative correlation.

    plt.title('Correlation Matrix for Selected Indicators')
    plt.show()

    return correlation_matrix

#PART FOUR: Data Visualization

#Visualizing the overall distrubutions of the Indicators
            
def plot_indicator_distribution_no_outliers(dataframe, indicators, countries):
    # Subset the dataframe with selected indicators and countries
    subset_program_dataframe = dataframe[dataframe['Country Code'].isin(countries) & dataframe['Indicator Code'].isin(indicators)]

    # Pivot the dataframe for easier plotting
    pivot_program_dataframe = subset_program_dataframe.melt(id_vars=['Country Code', 'Indicator Code', 'Country Name', 'Indicator Name'], 
                              var_name='Year', value_name='Value')

    # Convert 'Year' column to numeric (assuming it's in string format)
    pivot_program_dataframe['Year'] = pd.to_numeric(pivot_program_dataframe['Year'], errors='coerce')

    # Calculate the number of rows and columns for subplots
    num_indicators = len(indicators)
    num_cols = min(3, num_indicators)  # Set a maximum of 3 columns for better visualization
    num_rows = math.ceil(num_indicators / num_cols)

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))

    # Flatten the axes array if there's only one row
    axes = axes.flatten() if num_rows > 1 else [axes]

    # Plot histograms for each indicator
    for i, indicator in enumerate(indicators):
        indicator_name = pivot_program_dataframe[pivot_program_dataframe['Indicator Code'] == indicator]['Indicator Name'].iloc[0]
        indicator_program_dataframe = pivot_program_dataframe[pivot_program_dataframe['Indicator Code'] == indicator]
        sns.histplot(data=indicator_program_dataframe, x='Value', bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f'{indicator_name}')

    # Adjust layout
    plt.tight_layout()
    plt.show()

def visualize_and_tell_story(dataframe, indicators, countries):
    # Subset the dataframe with selected indicators and countries
    subset_program_dataframe = dataframe[dataframe['Country Code'].isin(countries) & dataframe['Indicator Code'].isin(indicators)]

    # Pivot the dataframe for easier plotting
    pivot_program_dataframe = subset_program_dataframe.melt(id_vars=['Country Code', 'Indicator Code', 'Country Name', 'Indicator Name'],
                              var_name='Year', value_name='Value')

    # Convert 'Year' column to numeric (assuming it's in string format)
    pivot_program_dataframe['Year'] = pd.to_numeric(pivot_program_dataframe['Year'], errors='coerce')

    # Set the style for better visualization
    sns.set(style="whitegrid")

    # Time series plot for selected indicators
    for country in countries:
        country_subset = pivot_program_dataframe[(pivot_program_dataframe['Country Code'] == country) & (pivot_program_dataframe['Indicator Code'].isin(indicators))]
        country_name = country_subset['Country Name'].iloc[0]  # Extract country name
        country_code = country_subset['Country Code'].iloc[0]  # Extract country code

        plt.figure(figsize=(12, 6))  # Adjust the figure size as needed

        # Plot time series using seaborn for better visualization
        sns.lineplot(data=country_subset, x='Year', y='Value', hue='Indicator Name', style='Indicator Name', markers=True, palette='Set2')

        plt.title(f'Selected Indicator Trends for {country} - {country_name}')
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.legend(title=f'{country} - {country_code}', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        plt.tight_layout()
        plt.show()  # Display each plot individually

        # Bar plot distribution for each indicator across countries
        for indicator in indicators:
            indicator_name = pivot_program_dataframe[pivot_program_dataframe['Indicator Code'] == indicator]['Indicator Name'].iloc[0]
            plt.figure(figsize=(10, 6))
            sns.barplot(data=pivot_program_dataframe[(pivot_program_dataframe['Indicator Code'] == indicator) & (pivot_program_dataframe['Country Code'].isin(countries))],
                        x='Country Name', y='Value', hue='Country Code', palette='Set2')  # Change palette to 'Set2'
            plt.title(f'{indicator_name} Across Countries')  # Update the title with the indicator name
            plt.xlabel('Country')
            plt.ylabel('Value')
            plt.legend(title='Country Code', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()  # Display each plot individually

def display_indicator_by_year(dataframe, indicators, countries, years):
    # Subset the dataframe with selected indicators, countries, and years
    subset_program_dataframe = dataframe[(dataframe['Country Code'].isin(countries)) & (dataframe['Indicator Code'].isin(indicators))]

    # Pivot the dataframe for easier display
    pivot_program_dataframe = subset_program_dataframe.melt(
        id_vars=['Country Code', 'Country Name', 'Indicator Code', 'Indicator Name'],
        var_name='Year',
        value_name='Value'
    )

    # Convert 'Year' column to numeric (assuming it's in string format)
    pivot_program_dataframe['Year'] = pd.to_numeric(pivot_program_dataframe['Year'], errors='coerce')

    # Filter the dataframe for the specified years
    pivot_program_dataframe_filtered = pivot_program_dataframe[pivot_program_dataframe['Year'].isin(years)]

    # Set the style for better visualization
    sns.set(style="whitegrid")

    # Create a bar plot for each indicator and year
    for indicator in indicators:
        indicator_name = pivot_program_dataframe[pivot_program_dataframe['Indicator Code'] == indicator]['Indicator Name'].iloc[0]

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=pivot_program_dataframe_filtered[pivot_program_dataframe_filtered['Indicator Code'] == indicator],
            x='Country Name',
            hue='Year',
            y='Value',
            palette='Paired'
        )
        plt.title(f'Indicator Values for {indicator_name} in Selected Countries')
        plt.xlabel('Country')
        plt.ylabel('Value')
        plt.legend(title='Year', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()  # Display each plot individually

            
# Load the data
filename = 'Data/dataset.csv'
program_dataframe, transposed_program_dataframe = process_worldbank_data(filename)

# Select a few countries for analysis
selected_countries = ['USA', 'CHN', 'IND', 'BRA', 'RUS']

# Explore most appearing indicators for each country
most_appearing_indicators = explore_most_appearing_indicators(program_dataframe, selected_countries)

# Print the results
for country, indicators in most_appearing_indicators.items():
    print(f"Most appearing indicators for {country}: {indicators}")

# Choose the top appearing indicators for further analysis
top_appearing_indicators = most_appearing_indicators[selected_countries[0]][:5]

# Run analysis
stats = explore_statistics(program_dataframe, top_appearing_indicators, selected_countries)
correlations = explore_correlations(program_dataframe, top_appearing_indicators, selected_countries)
visualize_and_tell_story(program_dataframe, top_appearing_indicators, selected_countries)
plot_indicator_distribution_no_outliers(program_dataframe, top_appearing_indicators, selected_countries)
selected_years = [1960, 2005, 2010, 2012, 2017, 2020]
display_indicator_by_year(program_dataframe, top_appearing_indicators, selected_countries, selected_years)
