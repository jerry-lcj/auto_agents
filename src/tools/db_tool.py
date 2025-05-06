#!/usr/bin/env python3
"""
Database Tool - Provides database connectivity and operations.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import logging
import os
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Base class for SQLAlchemy ORM models
Base = declarative_base()


class DatabaseTool:
    """
    Tool for interacting with databases using SQLAlchemy.
    
    This tool provides:
    1. Database connection management
    2. SQL query execution
    3. ORM operations
    4. Data extraction and transformation
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        echo: bool = False,
    ):
        """
        Initialize the DatabaseTool.
        
        Args:
            connection_string: SQLAlchemy connection URL
            echo: Whether to echo SQL statements to stdout
        """
        # Use provided connection string or get from environment variable
        self.connection_string = connection_string or os.getenv("DATABASE_URL")
        self.engine = None
        self.Session = None
        self.echo = echo
        
        # Initialize connection if connection string is provided
        if self.connection_string:
            self.connect(self.connection_string)
        else:
            logger.warning("No database connection string provided. Use connect() method to establish connection.")
    
    def connect(self, connection_string: str) -> bool:
        """
        Connect to a database.
        
        Args:
            connection_string: SQLAlchemy connection URL
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.connection_string = connection_string
            self.engine = create_engine(self.connection_string, echo=self.echo)
            self.Session = sessionmaker(bind=self.engine)
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    @contextmanager
    def session_scope(self):
        """
        Context manager for database sessions.
        
        Yields:
            SQLAlchemy session
        """
        if not self.Session:
            raise RuntimeError("Database not connected. Call connect() first.")
            
        session = self.Session()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a raw SQL query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of dictionaries representing result rows
        """
        if not self.engine:
            raise RuntimeError("Database not connected. Call connect() first.")
            
        params = params or {}
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                return [dict(row._mapping) for row in result]
        except SQLAlchemyError as e:
            logger.error(f"Query execution error: {e}")
            return []
            
    def create_tables(self):
        """Create all tables defined in SQLAlchemy models."""
        if not self.engine:
            raise RuntimeError("Database not connected. Call connect() first.")
            
        Base.metadata.create_all(self.engine)
    
    def get_table_names(self) -> List[str]:
        """
        Get list of tables in the database.
        
        Returns:
            List of table names
        """
        if not self.engine:
            raise RuntimeError("Database not connected. Call connect() first.")
            
        from sqlalchemy import inspect
        inspector = inspect(self.engine)
        return inspector.get_table_names()
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get schema information for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of dictionaries with column information
        """
        if not self.engine:
            raise RuntimeError("Database not connected. Call connect() first.")
            
        from sqlalchemy import inspect
        inspector = inspect(self.engine)
        return inspector.get_columns(table_name)