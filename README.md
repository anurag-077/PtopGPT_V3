# Real Estate Village Comparison - Django Application

This Django application replicates the functionality of the original Streamlit app for comparing real estate data between villages. It provides a modern web interface with Bootstrap styling and the same AI-powered analysis capabilities.

## Features

- **Village Comparison**: Compare real estate data between two villages
- **Multiple Categories**: Analyze Demand, Supply, Price, and Demography metrics
- **AI-Powered Analysis**: Uses OpenAI GPT-4o-mini for intelligent analysis
- **Modern UI**: Beautiful Bootstrap-based interface
- **Real-time Processing**: Asynchronous API calls for smooth user experience
- **Data Visualization**: Token usage tracking and source data display

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- SampleR.xlsx data file

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root with:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Place your data file**:
   - Copy your `SampleR.xlsx` file to the project root directory
   - The application will automatically create a `SampleR.pkl` file for faster loading

5. **Run Django migrations**:
   ```bash
   python manage.py migrate
   ```

6. **Start the development server**:
   ```bash
   python manage.py runserver
   ```

7. **Access the application**:
   Open your browser and go to `http://127.0.0.1:8000/`

## Usage

1. **Select Villages**: Choose two different villages from the dropdown menus
2. **Choose Categories**: Select one or more categories to analyze:
   - **Demand**: Sales data, carpet area consumed, project metrics
   - **Supply**: Total units, carpet area supplied, project counts
   - **Price**: Agreement prices, percentile rates, weighted averages
   - **Demography**: Pincode-wise and age-range-wise data
3. **Optional Query**: Enter a specific question or leave blank for default analysis
4. **Generate Analysis**: Click "Compare Villages" to get AI-powered insights
5. **View Results**: See the analysis, token usage, and source data

## Project Structure

```
real_estate_analyzer/
├── manage.py                 # Django management script
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── .env                     # Environment variables (create this)
├── SampleR.xlsx             # Your data file (add this)
├── real_estate_analyzer/    # Django project settings
│   ├── __init__.py
│   ├── settings.py          # Django settings
│   ├── urls.py              # Main URL configuration
│   ├── wsgi.py              # WSGI configuration
│   └── asgi.py              # ASGI configuration
├── analyzer/                # Main application
│   ├── __init__.py
│   ├── apps.py              # App configuration
│   ├── urls.py              # App URL patterns
│   └── views.py             # Business logic and API endpoints
└── templates/               # HTML templates
    └── analyzer/
        └── home.html        # Main interface template
```

## API Endpoints

- `GET /`: Main application interface
- `GET /get-villages/`: Get list of available villages
- `POST /compare/`: Compare villages and generate analysis

## Technical Details

### Data Processing
- Uses pandas for data manipulation
- FAISS vector store for semantic search
- HuggingFace embeddings for document similarity
- Fuzzy string matching for village name suggestions

### AI Integration
- OpenAI GPT-4o-mini for analysis generation
- Token counting for usage tracking
- Structured prompts for consistent analysis

### Frontend
- Bootstrap 5 for responsive design
- Modern gradient backgrounds and animations
- Real-time form validation
- Loading states and error handling

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**
   - Ensure your `.env` file exists and contains the correct API key
   - Restart the Django server after adding the `.env` file

2. **"No data for villages"**
   - Check that `SampleR.xlsx` is in the project root
   - Verify village names match exactly (case-insensitive)
   - Check the console for available village suggestions

3. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

4. **FAISS index errors**
   - The application automatically manages FAISS indices
   - If issues persist, delete any `faiss_index_*` folders and restart

### Performance Tips

- The first run will be slower as it creates the pickle file
- Subsequent runs use cached data for faster loading
- FAISS indices are created per comparison and cleaned up automatically

## Development

To modify the application:

1. **Add new categories**: Update `CATEGORY_MAPPING` in `views.py`
2. **Modify UI**: Edit `templates/analyzer/home.html`
3. **Change analysis logic**: Update the prompt template in `views.py`
4. **Add new features**: Extend the Django views and templates

## License

This project is for educational and demonstration purposes. Ensure you have proper licenses for any commercial use of the underlying technologies.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the console logs for detailed error messages
3. Ensure all dependencies and data files are properly configured 