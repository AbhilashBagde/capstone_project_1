# CAH → REH Conversion Financial Calculator

A comprehensive financial calculator for analyzing the conversion from Critical Access Hospital (CAH) to Rural Emergency Hospital (REH) status.

## Features

- **Staffing Analysis**: Model staffing cost reductions from 24/7 inpatient requirements
- **AFB Calculations**: Automated calculation of Medicare reimbursement impacts
- **Dynamic Projections**: 5-year financial projections with customizable assumptions
- **PDF Export**: Generate professional reports with charts and tables
- **CSV Export**: Export detailed financial data for further analysis
- **Multi-State Support**: Specialized calculations for facilities in AR, LA, NM, OK, and TX

## Deployment on Vercel

### Prerequisites
- Git repository (GitHub, GitLab, or Bitbucket)
- Vercel account (create at [vercel.com](https://vercel.com))

### Deployment Steps

1. **Push to Git Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/capstone_project_1.git
   git push -u origin main
   ```

2. **Deploy to Vercel**
   - Go to [vercel.com/new](https://vercel.com/new)
   - Import your Git repository
   - Select the repository
   - Click "Deploy"
   - Your site will be live at a URL like `capstone-project-1.vercel.app`

3. **Optional: Custom Domain**
   - After deployment, go to Settings → Domains
   - Add your custom domain
   - Update DNS records as instructed

### Local Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The application will be available at `http://localhost:3000`

## Project Structure

```
capstone_project_1/
├── CAH_REH_Calculator.html    # Main application (React embedded)
├── CAH_REH_Baselines.json     # Reference data
├── package.json               # Dependencies & scripts
├── vercel.json               # Vercel configuration
└── README.md                 # This file
```

## Data Source

The calculator loads baseline data from `CAH_REH_Baselines.json`. This file can be updated with new reference rates and historical data.

## Technology Stack

- **React 18** - UI framework (via CDN)
- **Babel** - JSX transpilation
- **Chart.js** - Financial visualizations
- **Tailwind CSS** - Styling
- **html2canvas & jsPDF** - PDF export
- **PapaParse** - CSV export

## Support

For issues or questions, please refer to the project documentation or contact the development team.

## License

MIT
