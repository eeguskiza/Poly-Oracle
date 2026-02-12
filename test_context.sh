#!/bin/bash

echo "========================================"
echo "Poly-Oracle Context Builder Test Suite"
echo "========================================"
echo ""

# Test 1: Ver markets disponibles
echo "Test 1: Listar markets disponibles"
echo "----------------------------------------"
./venv/bin/python cli.py markets --limit 3 2>&1 | grep -v "INFO\|DEBUG\|WARNING"
echo ""

# Test 2: Ver detalles de un market
echo "Test 2: Ver detalles del market 517310"
echo "----------------------------------------"
./venv/bin/python cli.py market 517310 2>&1 | grep -v "INFO\|DEBUG\|WARNING" | head -15
echo ""

# Test 3: Construir contexto completo para un market
echo "Test 3: Construir contexto completo para market 517310"
echo "----------------------------------------"
./venv/bin/python cli.py context 517310 2>&1 | grep -v "INFO\|DEBUG\|WARNING"
echo ""

# Test 4: Ver noticias relevantes para el market
echo "Test 4: Noticias relevantes para market 517310"
echo "----------------------------------------"
./venv/bin/python cli.py market-news 517310 2>&1 | grep -v "INFO\|DEBUG\|WARNING" | head -15
echo ""

echo "========================================"
echo "Tests completados!"
echo "========================================"
echo ""
echo "El contexto ha sido almacenado en ChromaDB en:"
echo "  - Colección 'news': Noticias relevantes"
echo "  - Colección 'market_context': Contexto completo del market"
