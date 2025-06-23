import React from 'react';

interface CitigroupLogoProps {
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const CitigroupLogo: React.FC<CitigroupLogoProps> = ({ 
  className = '', 
  size = 'md' 
}) => {
  const sizeClasses = {
    sm: 'h-6 w-24',
    md: 'h-8 w-32',
    lg: 'h-12 w-48'
  };

  return (
    <div className={`${sizeClasses[size]} ${className}`}>
      <svg
        viewBox="0 0 200 50"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full"
      >
        {/* Citigroup logo - simplified recreation */}
        {/* Red arc (iconic Citi umbrella) */}
        <path
          d="M8 25 C8 15, 18 8, 30 8 C42 8, 52 15, 52 25"
          stroke="#EE1C25"
          strokeWidth="8"
          fill="none"
          strokeLinecap="round"
        />
        {/* "citi" text */}
        <text
          x="8"
          y="40"
          fontSize="18"
          fontWeight="bold"
          fill="#004685"
          fontFamily="Arial, sans-serif"
        >
          citi
        </text>
        {/* "group" text */}
        <text
          x="60"
          y="40"
          fontSize="18"
          fontWeight="bold"
          fill="#004685"
          fontFamily="Arial, sans-serif"
        >
          group
        </text>
      </svg>
    </div>
  );
};
