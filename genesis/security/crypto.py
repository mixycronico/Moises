"""
Módulo de criptografía para el sistema Genesis.

Este módulo proporciona funcionalidades de cifrado y hashing para proteger
información sensible, como credenciales, claves API y datos de usuario.
"""

import base64
import hashlib
import hmac
import asyncio
import time
import logging
from typing import Tuple, Optional

# Verificar si tenemos la biblioteca de encriptación disponible
try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    from Crypto.Protocol.KDF import PBKDF2
    from Crypto.Util.Padding import pad, unpad
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Configuración de logging
logger = logging.getLogger(__name__)

class AESCipher:
    """
    Cifra y descifra datos sensibles usando AES-GCM o AES-CBC + HMAC
    con derivación segura de clave.
    """
    
    def __init__(self, key: str, use_gcm: bool = True, salt: Optional[bytes] = None):
        """
        Inicializar el cifrador AES.
        
        Args:
            key: Clave secreta
            use_gcm: Si se debe usar AES-GCM (más seguro que CBC)
            salt: Salt personalizado (opcional)
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError("La biblioteca PyCryptodome no está instalada. "
                             "Instálala con 'pip install pycryptodome'")
            
        self.bs = AES.block_size  # 16 bytes
        self.use_gcm = use_gcm  # AES-GCM por defecto, más seguro
        self.salt = salt if salt else get_random_bytes(16)  # Salt único por instancia
        
        # Derivar clave con PBKDF2 (más seguro que SHA-256 directo)
        self.key = PBKDF2(key.encode(), self.salt, dkLen=32, count=100000, 
                         prf=lambda p, s: hmac.new(p, s, hashlib.sha256).digest())
        self.hmac_key = PBKDF2(key.encode(), self.salt + b"hmac", dkLen=32, count=100000,
                              prf=lambda p, s: hmac.new(p, s, hashlib.sha256).digest())

    def encrypt(self, raw: str) -> str:
        """
        Cifra datos con AES-GCM o AES-CBC + HMAC.
        
        Args:
            raw: Datos a cifrar
            
        Returns:
            Datos cifrados en formato Base64
        """
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Datos a cifrar deben ser una cadena no vacía")

        if self.use_gcm:
            return self._encrypt_gcm(raw)
        else:
            return self._encrypt_cbc_hmac(raw)

    def decrypt(self, enc: str) -> str:
        """
        Descifra datos con AES-GCM o AES-CBC + HMAC.
        
        Args:
            enc: Datos cifrados en formato Base64
            
        Returns:
            Datos descifrados
        """
        if not isinstance(enc, str) or not enc.strip():
            raise ValueError("Datos cifrados deben ser una cadena no vacía")

        if self.use_gcm:
            return self._decrypt_gcm(enc)
        else:
            return self._decrypt_cbc_hmac(enc)

    def _encrypt_gcm(self, raw: str) -> str:
        """
        Cifrado autenticado con AES-GCM.
        
        Args:
            raw: Datos a cifrar
            
        Returns:
            Datos cifrados en formato Base64
        """
        iv = get_random_bytes(12)  # 12 bytes recomendado para GCM
        cipher = AES.new(self.key, AES.MODE_GCM, nonce=iv)
        ciphertext, tag = cipher.encrypt_and_digest(raw.encode('utf-8'))
        # Concatenar: IV (12) + Tag (16) + Ciphertext
        result = iv + tag + ciphertext
        return base64.b64encode(result).decode('utf-8')

    def _decrypt_gcm(self, enc: str) -> str:
        """
        Descifrado autenticado con AES-GCM.
        
        Args:
            enc: Datos cifrados en formato Base64
            
        Returns:
            Datos descifrados
        """
        try:
            enc_bytes = base64.b64decode(enc)
            iv = enc_bytes[:12]
            tag = enc_bytes[12:28]
            ciphertext = enc_bytes[28:]
            cipher = AES.new(self.key, AES.MODE_GCM, nonce=iv)
            plaintext = cipher.decrypt_and_verify(ciphertext, tag)
            return plaintext.decode('utf-8')
        except (ValueError, KeyError) as e:
            logger.error(f"Error al descifrar con GCM: {e}")
            raise ValueError("Datos corruptos o clave incorrecta")

    def _encrypt_cbc_hmac(self, raw: str) -> str:
        """
        Cifrado con AES-CBC y autenticación HMAC.
        
        Args:
            raw: Datos a cifrar
            
        Returns:
            Datos cifrados en formato Base64
        """
        raw_padded = pad(raw.encode('utf-8'), self.bs)
        iv = get_random_bytes(self.bs)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        encrypted = cipher.encrypt(raw_padded)
        # Calcular HMAC sobre IV + ciphertext
        hmac_value = hmac.new(self.hmac_key, iv + encrypted, hashlib.sha256).digest()
        # Concatenar: IV (16) + HMAC (32) + Ciphertext
        result = iv + hmac_value + encrypted
        return base64.b64encode(result).decode('utf-8')

    def _decrypt_cbc_hmac(self, enc: str) -> str:
        """
        Descifrado con AES-CBC y verificación HMAC.
        
        Args:
            enc: Datos cifrados en formato Base64
            
        Returns:
            Datos descifrados
        """
        try:
            enc_bytes = base64.b64decode(enc)
            iv = enc_bytes[:self.bs]
            hmac_value = enc_bytes[self.bs:self.bs + 32]
            ciphertext = enc_bytes[self.bs + 32:]
            # Verificar HMAC
            expected_hmac = hmac.new(self.hmac_key, iv + ciphertext, hashlib.sha256).digest()
            if not hmac.compare_digest(hmac_value, expected_hmac):
                raise ValueError("HMAC no coincide. Datos manipulados.")
            cipher = AES.new(self.key, AES.MODE_CBC, iv)
            decrypted = unpad(cipher.decrypt(ciphertext), self.bs)
            return decrypted.decode('utf-8')
        except (ValueError, KeyError) as e:
            logger.error(f"Error al descifrar con CBC+HMAC: {e}")
            raise ValueError("Datos corruptos o clave incorrecta")

    def get_salt(self) -> str:
        """
        Devuelve el salt en base64 para almacenamiento seguro.
        
        Returns:
            Salt en formato Base64
        """
        return base64.b64encode(self.salt).decode('utf-8')

    async def stress_test(self, num_messages: int):
        """
        Prueba de estrés con cifrado/descifrado masivo.
        
        Args:
            num_messages: Número de mensajes a cifrar
        """
        messages = [f"Test message {i}" for i in range(num_messages)]
        tasks_encrypt = [asyncio.to_thread(self.encrypt, msg) for msg in messages]
        start_time = time.time()
        
        # Cifrar concurrentemente
        encrypted = await asyncio.gather(*tasks_encrypt)
        logger.info(f"Cifrado de {num_messages} mensajes en {time.time() - start_time:.2f}s")

        # Descifrar concurrentemente
        tasks_decrypt = [asyncio.to_thread(self.decrypt, enc) for enc in encrypted]
        decrypted = await asyncio.gather(*tasks_decrypt, return_exceptions=True)
        elapsed = time.time() - start_time

        # Verificar integridad
        failures = sum(1 for orig, dec in zip(messages, decrypted) if isinstance(dec, Exception) or orig != dec)
        logger.info(f"Prueba completada en {elapsed:.2f}s. Fallos: {failures}/{num_messages}")

# Funciones de utilidad para hashing

def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    """
    Genera un hash seguro para una contraseña usando PBKDF2.
    
    Args:
        password: Contraseña a hashear
        salt: Salt personalizado (opcional)
        
    Returns:
        Tupla (hash_hex, salt_base64)
    """
    if not salt:
        salt = get_random_bytes(16) if CRYPTO_AVAILABLE else os.urandom(16)
        
    # Usar PBKDF2 si está disponible, o implementación personalizada si no
    if CRYPTO_AVAILABLE:
        key = PBKDF2(password.encode(), salt, dkLen=32, count=100000,
                   prf=lambda p, s: hmac.new(p, s, hashlib.sha256).digest())
    else:
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000, dklen=32)
        
    return key.hex(), base64.b64encode(salt).decode('utf-8')

def verify_password(password: str, stored_hash: str, stored_salt: str) -> bool:
    """
    Verifica si una contraseña coincide con un hash almacenado.
    
    Args:
        password: Contraseña a verificar
        stored_hash: Hash almacenado (formato hex)
        stored_salt: Salt almacenado (formato base64)
        
    Returns:
        True si la contraseña coincide, False en caso contrario
    """
    salt = base64.b64decode(stored_salt)
    calculated_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(calculated_hash, stored_hash)

def generate_api_key() -> Tuple[str, str]:
    """
    Genera un par de claves API (pública y secreta).
    
    Returns:
        Tupla (api_key, api_secret)
    """
    if CRYPTO_AVAILABLE:
        api_key = base64.b64encode(get_random_bytes(24)).decode('utf-8')
        api_secret = base64.b64encode(get_random_bytes(32)).decode('utf-8')
    else:
        import os
        api_key = base64.b64encode(os.urandom(24)).decode('utf-8')
        api_secret = base64.b64encode(os.urandom(32)).decode('utf-8')
        
    # Reemplazar caracteres no permitidos en URLs
    api_key = api_key.replace('+', '-').replace('/', '_').replace('=', '')
    api_secret = api_secret.replace('+', '-').replace('/', '_').replace('=', '')
    
    return api_key, api_secret