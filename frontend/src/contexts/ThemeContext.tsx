import React, { createContext, useContext, useEffect, useState } from 'react';

type Theme = 'light' | 'dark' | 'customer';

interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
  isCustomerTheme: boolean;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

interface ThemeProviderProps {
  children: React.ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [theme, setTheme] = useState<Theme>('light');

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') as Theme | null;
    if (savedTheme && ['light', 'dark', 'customer'].includes(savedTheme)) {
      applyTheme(savedTheme);
    }
  }, []);

  const applyTheme = (newTheme: Theme) => {
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Remove all theme classes first
    document.documentElement.classList.remove('dark', 'customer');
    document.body.classList.remove('dark', 'customer');
    
    if (newTheme === 'dark') {
      document.documentElement.classList.add('dark');
      document.body.classList.add('dark');
      document.body.style.backgroundColor = 'hsl(240 10% 3.9%)';
      document.body.style.color = 'hsl(0 0% 98%)';
      document.documentElement.style.backgroundColor = 'hsl(240 10% 3.9%)';
    } else if (newTheme === 'customer') {
      document.documentElement.classList.add('customer');
      document.body.classList.add('customer');
      document.body.style.backgroundColor = '#f8f9fa';
      document.body.style.color = '#004685';
      document.documentElement.style.backgroundColor = '#f8f9fa';
    } else {
      document.body.style.backgroundColor = 'hsl(0 0% 100%)';
      document.body.style.color = 'hsl(240 10% 3.9%)';
      document.documentElement.style.backgroundColor = 'hsl(0 0% 100%)';
    }
  };

  const handleSetTheme = (newTheme: Theme) => {
    applyTheme(newTheme);
  };

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    handleSetTheme(newTheme);
  };

  const value = {
    theme,
    setTheme: handleSetTheme,
    toggleTheme,
    isCustomerTheme: theme === 'customer',
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};
