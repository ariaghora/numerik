unit ndarray;

{$mode delphi}
{$H+}
{$modeSwitch advancedRecords}

interface

uses
  Classes, SysUtils, math;

type
  TLongVector = array of longint;
  TSingleVector = array of single;
  TUFunc = function(a: single): single;
  TBFunc<T> = function(a, b: T): T;

type
  TNDArray<T> = record
  private
    FIsContiguous: boolean;
    function GetNDims: integer;
    function GetSize: longint;
    function IndexToStridedOffset(Index: array of longint): longint;
    function OffsetToStrided(Offset: longint): longint;
    function ShapeToStrides(AShape: array of longint): TLongVector;
  public
    Data:  array of T;
    Shape: TLongVector;
    Strides: TLongVector;
    function Contiguous: TNDArray<T>;
    function Get(i: longint): T; overload;
    function Get(i: array of integer): T; overload;
    function Reshape(NewShape: array of longint): TNDArray<T>;
    property IsContiguous: boolean read FIsContiguous write FIsContiguous;
    property NDims: integer read GetNDims;
    property Size: longint read GetSize;

    class operator Add(A, B: TNDArray<T>): TNDArray<T>;
    class operator Subtract(A, B: TNDArray<T>): TNDArray<T>;
  end;

  TNDArrayArray<T> = array of TNDArray<T>;
  TBroadcastResult<T> = record
    A, B: TNDArray<T>;
  end;

  { The list goes on... }
  TSingleArray = TNDArray<single>;
  TLongArray   = TNDArray<longint>;
  TStringArray = TNDArray<string>;


  function BroadcastArrays<T>(A, B: TNDArray<T>): TBroadcastResult<T>;
  function CreateEmptyArray<T>(Contiguous: boolean = True): T;
  function CreateArray<T>(AData: array of T): TNDArray<T>;
  function CreateSingleArray(Data: array of single): TSingleArray;

  function AsStrided<T>(A: T; Shape, Strides: array of longint): T;
  function ApplyBFunc<T>(A, B: TNDArray<T>; BFunc: TBFunc<T>): TNDArray<T>;
  function DynArrayToVector(A: array of longint): TLongVector;

  procedure PrintArray<T>(A: T);

  { Compute the product of the items in Data }
  function Prod<T>(Data: array of T): T;
  function Reverse<T>(Data: T): T;
  function VectorsEqual<T>(A, B: T): boolean;

  procedure PrintVector<T>(Data: T);

  { Arithmetic function wrappers }
  function Add_S(a, b: single): single;
  function Multiply_S(a, b: single): single;
  function Subtract_S(a, b: single): single;

  function Add_String(a, b: string): string;

implementation

  function BroadcastArrays<T>(A, B: TNDArray<T>): TBroadcastResult<T>;
  var
    AdjShapeA, AdjShapeB, AdjStridesA, AdjStridesB: TLongVector;
    i, NewNDims: integer;
  begin
    NewNDims := Max(A.NDims, B.NDims);
    AdjShapeA := Reverse<TLongVector>(A.Shape);
    AdjShapeB := Reverse<TLongVector>(B.Shape);
    AdjStridesA := Reverse<TLongVector>(A.Strides);
    AdjStridesB := Reverse<TLongVector>(B.Strides);

    { Broadcast dimension is always max(A.NDims, B.NDims) }
    if A.NDims < NewNDims then
    begin
      SetLength(AdjShapeA, NewNDims);
      SetLength(AdjStridesA, NewNDims)
    end
    else
    begin
      SetLength(AdjShapeB, NewNDims);
      SetLength(AdjStridesB, NewNDims)
    end;

    for i := 0 to NewNDims - 1 do
      if (AdjShapeA[i] <> AdjShapeB[i]) then
      begin
        if (AdjShapeA[i] < AdjShapeB[i]) and (AdjShapeA[i] <= 1) then
        begin
          AdjStridesA[i] := 0;
          AdjShapeA[i] := AdjShapeB[i]
        end
        else if (AdjShapeB[i] < AdjShapeA[i]) and (AdjShapeB[i] <= 1) then
        begin
          AdjStridesB[i] := 0;
          AdjShapeB[i] := AdjShapeA[i]
        end
        else
          raise Exception.Create('Broadcasting failed.');
      end;

    Result.A := AsStrided<(TNDArray<T>)>(A, Reverse<TLongVector>(AdjShapeA),
                                            Reverse<TLongVector>(AdjStridesA));
    Result.B := AsStrided<(TNDArray<T>)>(B, Reverse<TLongVector>(AdjShapeB),
                                            Reverse<TLongVector>(AdjStridesB));
  end;

  function CreateEmptyArray<T>(Contiguous: boolean = True):T;
  begin
    Result.IsContiguous := Contiguous;
  end;

  function CreateArray<T>(AData: array of T): TNDArray<T>;
  var
    i: longint;
  begin
    Result := CreateEmptyArray<(TNDArray<T>)>(True);
    SetLength(Result.Data, Length(AData));
    for i := 0 to Length(AData) - 1 do
      Result.Data[i] := AData[i];
    SetLength(Result.Shape, 1);
    Result.Shape[0] := Length(AData);

    Result.Strides := Result.ShapeToStrides(Result.Shape);
  end;

  function CreateSingleArray(Data: array of single): TSingleArray;
  begin
    Result := CreateArray<single>(Data);
  end;

  function AsStrided<T>(A: T; Shape, Strides: array of longint): T;
  begin
    Result := CreateEmptyArray<T>(false);
    Result.Data := A.Data;
    Result.Shape := DynArrayToVector(Shape);
    Result.Strides := DynArrayToVector(Strides);
  end;

  function ApplyBFunc<T>(A, B: TNDArray<T>; BFunc: TBFunc<T>): TNDArray<T>;
  var
    i: longint;
    BcastResult: TBroadcastResult<T>;
  begin
    if not(VectorsEqual<TLongVector>(A.Shape, B.Shape)) then
    begin
      BcastResult := BroadcastArrays<T>(A, B);
      A := BcastResult.A;
      B := BcastResult.B;
    end;


    Result := CreateEmptyArray<(TNDArray<T>)>(True);
    SetLength(Result.Data, A.Size);
    Result.Shape := A.Shape;
    Result := Result.Reshape(A.Shape); // should be reset strides
    for i := 0 to A.Size - 1 do
      Result.Data[i] := BFunc(A.Get(i), B.Get(i));
  end;

  function DynArrayToVector(A: array of longint): TLongVector;
  var
    i: longint;
  begin
    Result := nil;
    SetLength(Result, Length(A));
    for i := 0 to High(A) do
      Result[i] := A[i];
  end;

  procedure PrintArray<T>(A: T);
  var
    r, c, i: longint;
  begin
    if A.NDims = 1 then
    begin
      for i := 0 to A.Size - 1 do
        Write(A.Get(i), ' ');
      WriteLn;
    end;

    if A.NDims = 2 then
    begin
      for r := 0 to A.Shape[0] - 1 do
      begin
        for c := 0 to A.Shape[1] - 1 do
        begin
          Write(A.Get([r, c]));
          if c < A.Shape[1] - 1 then Write(' ');
        end;
        WriteLn;
      end;
    end;
  end;

  function Prod<T>(Data: array of T): T;
  var
    x: T;
  begin
    Result := 1;
    for x in Data do
      Result := Result * x;
  end;

  function Reverse<T>(Data: T): T;
  var
    i: longint;
  begin
    Result := nil;
    SetLength(Result, Length(Data));
    For i := High(Data) downto 0 do
      Result[High(Data) - i] := Data[i];
  end;

  function VectorsEqual<T>(A, B: T): boolean;
  var
    i: longint;
  begin
    if (Length(A) <> Length(B)) then Exit(false);
    Result := True;
    for i := 0 to High(A) do
      Result := Result and (A[i] = B[i]);
  end;

  procedure PrintVector<T>(Data: T);
  var
    i: longint;
  begin
    for i := 0 to Length(Data) - 1 do
    begin
      Write(Data[i]);
      if i < Length(Data) - 1 then
        Write(', ');
    end;
    WriteLn;
  end;

  function TNDArray<T>.ShapeToStrides(AShape: array of longint): TLongVector;
  var
    k, j, prod: integer;
  begin
    Result := nil;

    SetLength(Result, Length(AShape));
    for k := 0 to Length(Result) - 1 do
    begin
      prod := 1;
      for j := k + 1 to Length(Result) - 1 do
        prod := prod * AShape[j];
      Result[k] := prod;
    end;
  end;

  function TNDArray<T>.OffsetToStrided(Offset: longint): longint;
  var
    r, c: longint;
  begin
    if NDims > 2 then
      raise ENotImplemented.Create('Item access with not implemented yet with NDims > 2.');

    if NDims = 1 then Exit(Offset);
    if NDims = 2 then
    begin
      r := Offset div Shape[1];
      c := Offset mod Shape[1];
      Exit(IndexToStridedOffset([r, c]));
    end;
  end;

  function TNDArray<T>.GetNDims: integer;
  begin
    Exit(Length(Shape));
  end;

  function TNDArray<T>.GetSize: longint;
  begin
    Exit(Prod<longint>(Shape));
  end;

  function TNDArray<T>.IndexToStridedOffset(Index: array of longint): longint;
  var
    i: integer;
  begin
    Result := 0;
    for i := 0 to Length(Index) - 1 do
      Result := Result + (Strides[i] * Index[i]);
  end;

  function TNDArray<T>.Contiguous: TNDArray<T>;
  begin
    if IsContiguous then Exit(self);
  end;

  function TNDArray<T>.Get(i: longint): single; overload;
  begin
    if IsContiguous then Exit(Data[i]);
    Exit(Data[OffsetToStrided(i)]);
  end;

  function TNDArray<T>.Get(i: array of integer): single; overload;
  begin
    Assert(length(i) = length(Shape), 'Index dimension does not match array dimension.');
    Exit(Data[IndexToStridedOffset(i)]);
  end;

  function TNDArray<T>.Reshape(NewShape: array of longint): TNDArray<T>;
  var
    i: integer;
  begin
    Assert((Prod<longint>(NewShape)) = (Prod<longint>(Shape)), 'Impossible reshape.');
    SetLength(Shape, Length(NewShape));
    for i := 0 to Length(NewShape) - 1 do
      Shape[i] := NewShape[i];
    Strides := ShapeToStrides(Shape);
    Exit(self);
  end;

  class operator TNDArray<T>.Add(A, B: TNDArray<T>): TNDArray<T>;
  begin
    if TypeInfo(T) = TypeInfo(single) then
      Exit(ApplyBFunc<T>(A, B, @Add_S))
    else if TypeInfo(T) = TypeInfo(string) then
      Exit(ApplyBFunc<T>(A, B, @Add_String))
    else
      raise ENotImplemented.Create('Add operator is not implemented for this datatype');
  end;

  class operator TNDArray<T>.Subtract(A, B: TNDArray<T>): TNDArray<T>;
  begin
    if TypeInfo(T) = TypeInfo(single) then
      Result := ApplyBFunc<T>(A, B, @Subtract_S)
    else
      raise ENotImplemented.Create('Add operator is not implemented for this datatype');
  end;

  function Add_S(a, b: single): single;
  begin
    Exit(a + b);
  end;

  function Multiply_S(a, b: single): single;
  begin
    Exit(a * b);
  end;

  function Subtract_S(a, b: single): single;
  begin
    Exit(a - b);
  end;

  function Add_String(a, b: string): string;
  begin
    Exit(a + b);
  end;

end.

