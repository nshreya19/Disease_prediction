
CREATE TABLE user (
    PatientID INT AUTO_INCREMENT PRIMARY KEY,
    Name VARCHAR(255) NOT NULL,
    Email VARCHAR(255) UNIQUE NOT NULL,
    PhoneNumber VARCHAR(15) NOT NULL,
    Age INT NOT NULL,
    Gender ENUM('male', 'female', 'other') NOT NULL,
    Location VARCHAR(255) NOT NULL,
    Password VARCHAR(255) NOT NULL
    
);


