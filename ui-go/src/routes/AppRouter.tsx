import { BrowserRouter, Routes, Route } from 'react-router-dom';
import DashboardPage from '../features/dashboard/DashboardPage';
import DatasetListPage from '../features/datasets/DatasetListPage';
import DatasetDetailPage from '../features/datasets/DatasetDetailPage';
import JobListPage from '../features/jobs/JobListPage';
import JobNewPage from '../features/jobs/JobNewPage';
import JobDetailPage from '../features/jobs/JobDetailPage';
import SettingsPage from '../features/settings/SettingsPage';
import MainLayout from '../layouts/MainLayout';

export default function AppRouter() {
  return (
    <BrowserRouter>
      <MainLayout>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/datasets" element={<DatasetListPage />} />
          <Route path="/datasets/:datasetName" element={<DatasetDetailPage />} />
          <Route path="/jobs" element={<JobListPage />} />
          <Route path="/jobs/new" element={<JobNewPage />} />
          <Route path="/jobs/:jobID" element={<JobDetailPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </MainLayout>
    </BrowserRouter>
  );
} 