import Sidebar from '@/components/Sidebar';
import { ThemeProvider } from '@/components/ThemeProvider';
import ConfirmModal from '@/components/ConfirmModal';
import SampleImageModal from '@/components/SampleImageModal';
import DocModal from '@/components/DocModal';
import { Suspense } from 'react';
import AuthWrapper from '@/components/AuthWrapper';

export default function MainLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <ThemeProvider>
        <AuthWrapper authRequired={false}>
          <div className="flex h-screen bg-gray-950">
            <Sidebar />
            <main className="flex-1 overflow-auto bg-gray-950 text-gray-100 relative">
              <Suspense>{children}</Suspense>
            </main>
          </div>
        </AuthWrapper>
      </ThemeProvider>
      <ConfirmModal />
      <SampleImageModal />
      <DocModal />
    </>
  );
}
