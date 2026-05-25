-- Crear la base de datos
CREATE DATABASE iF NOT EXISTS NeoMarketDW;
USE NeoMarketDW;

-- =========================
-- TABLAS DE DIMENSIONES
-- =========================

-- Tiempo
CREATE TABLE iF NOT EXISTS Tiempo (
    ID_Tiempo INT AUTO_INCREMENT PRIMARY KEY,
    Fecha DATE NOT NULL,
    Dia INT,
    Mes INT,
    Anio INT,
    Dia_Semana VARCHAR(15),
    Trimestre INT
);

-- Producto
CREATE TABLE iF NOT EXISTS Producto (
    ID_Producto INT AUTO_INCREMENT PRIMARY KEY,
    Nombre_Producto VARCHAR(100) NOT NULL,
    Categoria VARCHAR(50),
    Subcategoria VARCHAR(50),
    Marca VARCHAR(50)
);

-- Cliente
CREATE TABLE iF NOT EXISTS Cliente (
    ID_Cliente INT AUTO_INCREMENT PRIMARY KEY,
    Nombre VARCHAR(100),
    Edad INT,
    Genero VARCHAR(20),
    Localidad VARCHAR(50),
    Nivel_Socioeconomico VARCHAR(20)
);

-- Tienda
CREATE TABLE iF NOT EXISTS Tienda (
    ID_Tienda INT AUTO_INCREMENT PRIMARY KEY,
    Nombre_Tienda VARCHAR(100) NOT NULL,
    Localidad VARCHAR(50),
    Tamano_Tienda VARCHAR(50),
    Fecha_Apertura DATE
);

-- =========================
-- TABLA DE HECHOS
-- =========================

CREATE TABLE iF NOT EXISTS Ventas (
    ID_Venta INT AUTO_INCREMENT PRIMARY KEY,
    ID_Tiempo INT,
    ID_Producto INT,
    ID_Cliente INT,
    ID_Tienda INT,
    Cantidad INT NOT NULL,
    Precio_Unitario INT NOT NULL,
    Descuento INT DEFAULT 0,
    Total_Venta INT NOT NULL,

    -- Relaciones (FKs)
    FOREIGN KEY (ID_Tiempo) REFERENCES Tiempo(ID_Tiempo),
    FOREIGN KEY (ID_Producto) REFERENCES Producto(ID_Producto),
    FOREIGN KEY (ID_Cliente) REFERENCES Cliente(ID_Cliente),
    FOREIGN KEY (ID_Tienda) REFERENCES Tienda(ID_Tienda)
);

SELECT * FROM Cliente;
SELECT * FROM Producto;
SELECT * FROM Tienda;
SELECT * FROM Tiempo;
SELECT * FROM Ventas;