# How to use

---

## 1. Clonar el repositorio desde Git

```bash
git clone https://github.com/SamuelCastroUPTC/BaseDeDatosRiskAnalysis.git
cd repositorio
```

---

## 2. Crear el entorno virtual (`venv`)

Ubicado dentro de la carpeta del proyecto:

```bash
python3 -m venv .venv
```
---

## 3. Activar el entorno virtual

**Linux / macOS (bash o zsh):**
```bash
source .venv/bin/activate
```

**Windows (cmd):**
```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

Sabrás que está activo porque tu terminal mostrará el prefijo `(.venv)` antes del prompt.

---

## 4. Actualizar `pip` (opcional pero recomendado)

```bash
pip install --upgrade pip
```

---

## 5. Instalar las dependencias del proyecto

```bash
pip install -r requirements.txt
```


---

## 6. Correr el proyecto

```bash
python main.py
```

---
